from flask import Flask, render_template, send_file
import tempfile
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired
from wtforms import FileField, SubmitField, IntegerField
from wtforms.validators import DataRequired, NumberRange

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def rgb_to_hex(color):
    color = [round(value) for value in color]
    return "#{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])


def extract_colors(image: Image, number_of_clusters):
    array = np.array(image)
    array = np.reshape(array, (-1, 3))

    unique_colors = np.unique(array, axis=0)
    number_of_unique_colors = len(unique_colors)

    if number_of_unique_colors < number_of_clusters:
        number_of_clusters = number_of_unique_colors

    kmeans = KMeans(n_clusters=number_of_clusters, n_init='auto')
    kmeans.fit(array)
    labels = kmeans.labels_
    labels = list(labels)
    centroid = list(kmeans.cluster_centers_)

    colors = {}
    for i in range(len(centroid)):
        j = labels.count(i)
        j = round(100*j / (len(labels)), 4)
        colors[rgb_to_hex(centroid[i])] = f"{j} %"

    return colors


def allowed_file(file_name):
    file_extension = file_name[file_name.rfind("."):].lower()
    return file_extension in ALLOWED_EXTENSIONS


class UploadForm(FlaskForm):
    file_input = FileField('Upload', validators=[FileRequired()])
    clusters = IntegerField(label='Number of Clusters', validators=[DataRequired(), NumberRange(min=1, max=20)])
    submit = SubmitField('Upload')


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "SomeSecretKey"


@app.route("/", methods=["GET", "POST"])
def home():
    image_url = None
    colors = None
    form = UploadForm(clusters=10)
    if form.validate_on_submit():
        file = form.file_input.data
        if file and allowed_file(file.filename):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                content = file.read()
                temp_file.write(content)
                image_url = temp_file.name
                image = Image.open(file.stream)
                colors = extract_colors(image, form.clusters.data)
    return render_template("index.html", image_path=image_url, colors=colors, form=form)


@app.route('/temp_image/<path:temp_file_path>')
def serve_temp_image(temp_file_path):
    return send_file(temp_file_path, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run()
