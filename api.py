from predictor import DigitPredictor
from flask import Flask, request, jsonify 
import utils
app = Flask(__name__)

predictor = DigitPredictor()


@app.route('/')
def index():
    return "hello WOrld"

@app.route('/v1/predict',methods=['GET','POST'])
def predict():
    image = _load_image()
    # with tf.Session().graph.as_default() as _:
    pred = predictor.predict_on_image(image)

    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred)})

def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return utils.read_b64_image(data['image'], grayscale=True)
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return utils.read_image(image_url, grayscale=True)
    raise ValueError('Unsupported HTTP method')

def main():
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
