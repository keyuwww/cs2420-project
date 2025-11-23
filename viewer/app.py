from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
from pathlib import Path

app = Flask(__name__)

# Configuration
RESULTS_FILE = os.environ.get('RESULTS_FILE', '../output.jsonl')
IMAGE_DIR = os.environ.get('IMAGE_DIR', '../coco_val_sample-SPECIFIC')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """Load all results from JSONL file"""
    results = []
    try:
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        return jsonify({"error": "Results file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(results)

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    """Serve images from the image directory"""
    # Extract just the filename from the full path
    base_filename = os.path.basename(filename)
    return send_from_directory(os.path.abspath(IMAGE_DIR), base_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
