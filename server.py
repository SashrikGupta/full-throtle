from flask import Flask, request, jsonify
from flask_cors import CORS

from google import genai
genai_client = genai.Client(api_key="AIzaSyCSZUI-z4ogIKeB7uDgy-WmHXDPJ0Prf-8")

from groq import Groq
groq_client = Groq(api_key="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")

app = Flask(__name__)
CORS(app)

@app.route('/gemini', methods=['GET'])
def gemini():
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt parameter is required'}), 400
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/deepseek', methods=['GET'])
def deepseek():
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt parameter is required'}), 400
    try:
        completion = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=32768,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/llama3', methods=['GET'])
def llama3():
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt parameter is required'}), 400
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=32768,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/llama4', methods=['GET'])
def llama4():
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt parameter is required'}), 400
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
