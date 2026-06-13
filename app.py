import os

from flask import Flask, jsonify, render_template, request

from gpt_lib import DEFAULT_MODEL_PATH, load_model


app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", str("output/MiniGPT.pth")) # or use MeduimGPT.pth
_loaded_model = None


def get_loaded_model():
    """Lazy-load the model once per Flask process."""
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}.")
    return _loaded_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Voer eerst een prompt in."}), 400

        loaded_model = get_loaded_model()
        result = loaded_model.predict(
            prompt,
            max_new_tokens=_bounded_int(payload.get("max_new_tokens"), 80, 1, 300),
            temperature=_bounded_float(payload.get("temperature"), 0.9, 0.1, 2.0),
            repetition_penalty=_bounded_float(
                payload.get("repetition_penalty"),
                1.2,
                1.0,
                3.0,
            ),
            top_p=_bounded_float(payload.get("top_p"), 0.9, 0.05, 1.0),
        )

        return jsonify({"result": result, "model_path": MODEL_PATH})

    except Exception as exc:
        print("ERROR:", str(exc))
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    return generate()


def _bounded_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bounded_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
