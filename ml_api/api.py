from flask import Flask, jsonify
from vectorize import get_feature_vector
import scipy.ndimage
from PIL import Image
from random import random

app = Flask(__name__)
imn = np.load('image_mean.npy')
isd = np.load('image_std.npy')


@app.route('/api/regress', methods=['POST']):
    caption = request.form['caption']
    followers = int(request.form['followers'])
    hour = int(request.form['hour'])
    day = request.form['day']
    img_file = request.file['file']
    day_vector = [0, 0, 0, 0, 0, 0, 0]
    if day == 'Sunday':
        day_vector[0] = 1
    elif day == 'Monday':
        day_vector[1] = 1
    elif day == 'Tuesday':
        day_vector[2] = 1
    elif day == 'Wednesday':
        day_vector[3] = 1
    elif day == 'Thursday':
        day_vector[4] = 1
    elif day == 'Friday':
        day_vector[5] = 1
    elif day == 'Saturday':
        day_vector[6] = 1
    feature_vector = get_feature_vector(caption, day_vector, hour, followers, 0.5)
    wordvec = vectors[:384]
    vectors = feature_vector[384:393]
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

    values = (image, wordvec, vectors)

    # Then you want to call TensorFlow API on (image, wordvec, vectors)
    # For now, let's do the best prediction algorithm
    return jsonify({'likes': random() * followers})



if __name__ == '__main__':
    app.run(debug=True)
