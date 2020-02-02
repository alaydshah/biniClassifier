import flask
from flask import Flask
import time
import os
import numpy as np
import tensorflow as tf
import urllib
import urllib.request

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World"
class Classifier:
    modelPath = None
    labelPath = None
    sess = None
    softmax_tensor = None

    def __init__(self, modelPath, labelPath):
        self.modelPath = modelPath
        self.labelPath = labelPath
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(modelPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')

    def get_image_labels(self, imagePath):
        answer = None

        req = urllib.request.Request(imagePath)
        response = urllib.request.urlopen(req)

        image_data = response.read()
#        print(image_data)

        predictions = self.sess.run(self.softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
#        print(predictions)
        top_k = predictions.argsort()[-6:][::-1] # Getting top 5 predictions
#        print("TopK:")
#        print(top_k)
        f = open(self.labelPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
#        print(labels)
        answers = []
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            answers.append({"type": human_string, "score" : score.item() })
        return answers

def getRecyclingLabel(labels):
	for label in labels:
		# If we have a 45% confidence it's recyclable
		if label["score"] > .55:
			return label["type"]
	return None


@app.route("/predict", methods = ["POST"])
def predict():

    data = {}
    data["success"] = False
    results = None

    classifier = Classifier(os.path.abspath('classifier/trained_graph.pb'),
                            os.path.abspath('classifier/output_labels.txt'))

    # Check if image was properly sent to our endpoint
    if flask.request.method == "POST":
        print("Request Form:")
        print(flask.request.form)
        imguri = flask.request.form['imageUri']
        print("Image Path:")
        print(imguri)
        if flask.request.form["imageUri"]:
            image_path = flask.request.form["imageUri"]
            labels = classifier.get_image_labels(image_path)

            if not labels:
                data["predictions"] = results
                return data

            selectedLabel = getRecyclingLabel(labels)

            is_trash = selectedLabel == None

            timestamp = int(time.time())  # Seconds since UNIX epoch

            results = {'labels': labels, 'isTrash': is_trash, 'timestamp': timestamp,
                       'recyclingLabel': selectedLabel}

    if results:
        data["success"] = True

    data["predictions"] = results
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
