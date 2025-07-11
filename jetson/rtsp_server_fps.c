#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h> // Required for GMutex and GstClockTime

#define DEFAULT_RTSP_PORT "8554"

static char *port = (char *) DEFAULT_RTSP_PORT;

// Structure to hold FPS tracking data.
// This will be passed as user_data to the pad probe callback.
typedef struct {
  guint frame_count;      // Counter for frames processed in the current interval
  GstClockTime last_time; // Timestamp of the last FPS calculation
  gdouble current_fps;    // Stores the calculated FPS
  GMutex mutex;           // Mutex for thread-safe access to tracker data
} FpsTracker;

// Global instance of FpsTracker. Since the media factory is set to shared,
// all clients will use the same pipeline, and thus a single tracker is sufficient.
static FpsTracker fps_tracker = { 0, 0, 0.0 };

/**
 * @brief Pad probe callback function to calculate and print FPS.
 *
 * This function is called every time a buffer passes through the probed pad.
 * It increments a frame counter and, if enough time has passed (e.g., 1 second),
 * calculates the FPS and prints it.
 *
 * @param pad The GstPad the probe is attached to.
 * @param info Information about the probe (e.g., buffer, event, query).
 * @param user_data A pointer to the FpsTracker structure.
 * @return GST_PAD_PROBE_OK to allow the buffer to continue processing.
 */
static GstPadProbeReturn
fps_probe_cb (GstPad * pad, GstPadProbeInfo * info, gpointer user_data)
{
  FpsTracker *tracker = (FpsTracker *) user_data;

  // Only process buffer types. Other types like events or queries are ignored for FPS.
  if (GST_PAD_PROBE_INFO_TYPE (info) & GST_PAD_PROBE_TYPE_BUFFER) {
    // Lock the mutex to ensure thread-safe access to the tracker data.
    // Probes are typically called on the streaming thread.
    g_mutex_lock (&tracker->mutex);

    tracker->frame_count++; // Increment the frame counter

    // Get the current system clock time.
    GstClockTime now = gst_clock_get_time (gst_system_clock_obtain ());

    // Initialize last_time on the first buffer.
    if (tracker->last_time == 0) {
      tracker->last_time = now;
    } else {
      // Calculate elapsed time in seconds.
      gdouble elapsed_seconds = (gdouble) (now - tracker->last_time) / GST_SECOND;

      // Update FPS every 1 second (or more frequently if desired).
      if (elapsed_seconds >= 1.0) {
        tracker->current_fps = tracker->frame_count / elapsed_seconds;
        g_print ("Current FPS: %.2f\n", tracker->current_fps); // Print the calculated FPS
        tracker->frame_count = 0; // Reset frame counter for the next interval
        tracker->last_time = now;   // Reset last time for the next interval
      }
    }
    // Unlock the mutex.
    g_mutex_unlock (&tracker->mutex);
  }

  return GST_PAD_PROBE_OK; // Allow the buffer to continue its journey through the pipeline
}

// --- Custom GstRTSPMediaFactory Implementation ---
// We need a custom factory to override the create_element method,
// which is where we can access the newly created GStreamer pipeline
// and add our pad probe.

typedef struct _MyMediaFactory MyMediaFactory;
typedef struct _MyMediaFactoryClass MyMediaFactoryClass;

struct _MyMediaFactory {
  GstRTSPMediaFactory parent;
};

struct _MyMediaFactoryClass {
  GstRTSPMediaFactoryClass parent_class;
};

// G_DEFINE_TYPE macro helps in boilerplate code for GObject type registration.
G_DEFINE_TYPE (MyMediaFactory, my_media_factory, GST_TYPE_RTSP_MEDIA_FACTORY);

/**
 * @brief Overridden create_element method for MyMediaFactory.
 *
 * This method is called by the RTSP server to create the GStreamer pipeline
 * based on the launch line. After the pipeline is created, we find the
 * 'pay0' element (rtph264pay in your case) and attach our FPS probe to its sink pad.
 *
 * @param factory The custom media factory instance.
 * @param url The RTSP URL for which the pipeline is being created.
 * @return The created GstElement (pipeline) or NULL on failure.
 */
static GstElement *
my_media_factory_create_element (GstRTSPMediaFactory * factory, const GstRTSPUrl * url)
{
  GstElement *pipeline;
  GstElement *payloader;
  GstPad *sink_pad;

  // Call the parent's create_element to get the base pipeline.
  // This uses the launch line set on the factory to build the pipeline.
  pipeline = GST_RTSP_MEDIA_FACTORY_CLASS (my_media_factory_parent_class)->create_element (factory, url);

  if (!pipeline) {
    g_printerr ("Error: Failed to create pipeline from launch line.\n");
    return NULL;
  }

  // Find the payloader element by its name.
  // In your launch line, 'rtph264pay' is named 'pay0'.
  payloader = gst_bin_get_by_name (GST_BIN (pipeline), "pay0");
  if (!payloader) {
    g_printerr ("Error: Could not find 'pay0' element in the pipeline. Cannot attach FPS probe.\n");
    gst_object_unref (pipeline); // Release the pipeline if payloader not found
    return NULL;
  }

  // Get the sink pad of the payloader element.
  // This is where encoded video buffers arrive before being RTP packaged.
  sink_pad = gst_element_get_static_pad (payloader, "sink");
  if (!sink_pad) {
    g_printerr ("Error: Could not get sink pad of 'pay0'. Cannot attach FPS probe.\n");
    gst_object_unref (payloader); // Release the payloader
    gst_object_unref (pipeline);  // Release the pipeline
    return NULL;
  }

  // Add the pad probe to the sink pad.
  // The fps_probe_cb will be called for every buffer.
  // We pass the address of our global fps_tracker as user_data.
  gst_pad_add_probe (sink_pad, GST_PAD_PROBE_TYPE_BUFFER, fps_probe_cb, &fps_tracker, NULL);

  // Unreference the pad and element. The pipeline still holds a reference to them.
  gst_object_unref (sink_pad);
  gst_object_unref (payloader);

  return pipeline; // Return the created pipeline
}

/**
 * @brief Initialization function for MyMediaFactory instances.
 *
 * This function is called when a new instance of MyMediaFactory is created.
 * We use it to initialize the mutex for our FPS tracker.
 *
 * @param factory The MyMediaFactory instance being initialized.
 */
static void
my_media_factory_init (MyMediaFactory * factory)
{
  // Initialize the mutex for thread-safe access to fps_tracker.
  g_mutex_init (&fps_tracker.mutex);
}

/**
 * @brief Class initialization function for MyMediaFactory.
 *
 * This function is called once when the MyMediaFactory type is registered.
 * We override the create_element method of the parent GstRTSPMediaFactoryClass.
 *
 * @param klass The MyMediaFactoryClass being initialized.
 */
static void
my_media_factory_class_init (MyMediaFactoryClass * klass)
{
  GstRTSPMediaFactoryClass *gstrtspmediafactory_class = GST_RTSP_MEDIA_FACTORY_CLASS (klass);
  // Assign our custom create_element function to override the parent's.
  gstrtspmediafactory_class->create_element = my_media_factory_create_element;
}

// --- Main application code ---

static GOptionEntry entries[] = {
  {"port", 'p', 0, G_OPTION_ARG_STRING, &port,
      "Port to listen on (default: " DEFAULT_RTSP_PORT ")", "PORT"},
  {NULL}
};

int
main (int argc, char *argv[])
{
  GMainLoop *loop;
  GstRTSPServer *server;
  GstRTSPMountPoints *mounts;
  MyMediaFactory *factory; // Declare our custom factory type
  GOptionContext *optctx;
  GError *error = NULL;

  // Initialize GLib and GStreamer.
  optctx = g_option_context_new ("<launch line> - Test RTSP Server, Launch\n\n"
      "Example: \"( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )\"");
  g_option_context_add_main_entries (optctx, entries, NULL);
  g_option_context_add_group (optctx, gst_init_get_option_group ());
  if (!g_option_context_parse (optctx, &argc, &argv, &error)) {
    g_printerr ("Error parsing options: %s\n", error->message);
    g_option_context_free (optctx);
    g_clear_error (&error);
    return -1;
  }
  g_option_context_free (optctx);

  // Ensure a launch line argument is provided.
  if (argc < 2) {
    g_printerr ("Usage: %s <launch line>\n", argv[0]);
    return -1;
  }

  // Create a new GMainLoop to run the server.
  loop = g_main_loop_new (NULL, FALSE);

  /* Create a new RTSP server instance */
  server = gst_rtsp_server_new ();
  // Set the port for the RTSP server.
  g_object_set (server, "service", port, NULL);

  /* Get the mount points for this server.
   * Mount points define the URI paths where streams are available. */
  mounts = gst_rtsp_server_get_mount_points (server);

  /* Create an instance of our custom media factory.
   * Instead of gst_rtsp_media_factory_new(), we use g_object_new()
   * with our custom type. */
  factory = g_object_new (my_media_factory_get_type (), NULL);

  // Set the GStreamer launch line for the factory.
  // This line defines the pipeline that will generate the video stream.
  gst_rtsp_media_factory_set_launch (GST_RTSP_MEDIA_FACTORY (factory), argv[1]);
  // Set the factory to shared, meaning all clients will share the same pipeline instance.
  gst_rtsp_media_factory_set_shared (GST_RTSP_MEDIA_FACTORY (factory), TRUE);

  /* Attach our custom factory to the "/test" URI path. */
  gst_rtsp_mount_points_add_factory (mounts, "/test", GST_RTSP_MEDIA_FACTORY (factory));

  /* Release the reference to the mount points object.
   * The server still holds a reference. */
  g_object_unref (mounts);

  /* Attach the server to the default GLib main context.
   * This makes the server start listening for incoming connections. */
  gst_rtsp_server_attach (server, NULL);

  /* Print the RTSP URL where the stream is available. */
  g_print ("Stream ready at rtsp://127.0.0.1:%s/test\n", port);

  /* Start the GLib main loop. This will keep the server running
   * and process all GStreamer and GLib events. */
  g_main_loop_run (loop);

  // Clean up the mutex when the application exits.
  g_mutex_clear (&fps_tracker.mutex);

  // Clean up GStreamer and GLib resources (though typically not reached in a server loop).
  gst_object_unref (server);
  g_main_loop_unref (loop);

  return 0;
}