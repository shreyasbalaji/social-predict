from flask import Flask, jsonify
from vectorize import get_feature_vector
import scipy.ndimage
from PIL import Image

app = Flask(__name__)
imn = np.load('image_mean.npy')
isd = np.load('image_std.npy')


@app.route('/api/regress', methods=['POST']):
    caption = request.form['caption']
    followers = request.form['followers']
    hour = 22
    day = [1, 0, 0, 0, 0, 0]
    img_file = request.file['image']
    feature_vector = get_feature_vector(caption, day, hour, followers, 0.5)
    if img_file:
        filename = secure_filename(img_file.filename)
        savepath = os.path.join('tmpdir', filename)
        outpath = os.path.join('tmpdir', 'ftest.jpg')

        img_file.save(savepath)
        img = Image.open(savepath)
        img.resize((300, 300), Image.ANTIALIAS)
        img.save(outpath)
        os.path.remove(savepath)

        image = (scipy.ndimage.imread(fname) - imn) / isd

    # Then you want to call TensorFlow API on (image, feature_vector)


if __name__ == '__main__':
    app.run(debug=True)
