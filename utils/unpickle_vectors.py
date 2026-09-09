import pickle

def unpickle_file(filename):
    
    try:
        with open(filename, 'rb') as file:
            unpickled_object = pickle.load(file)
        return unpickled_object
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error unpickling file '{filename}': {e}")
        return None

file_to_unpickle = 'D:/Projects/6GRescue-FaceRecog/jetson/face_features.pkl'
unpickled_data = unpickle_file(file_to_unpickle)

if unpickled_data:
    print("Unpickled data:", unpickled_data)