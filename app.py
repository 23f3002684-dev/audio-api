from flask import Flask, request, jsonify
import base64
import numpy as np
import io

app = Flask(__name__)

def decode_audio(audio_base64):
    """Decode base64 audio data"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert bytes to numpy array
        # Assuming 16-bit PCM audio
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalize to float between -1 and 1
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        return audio_float
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_features(audio_data):
    """Extract statistical features from audio"""
    
    if audio_data is None or len(audio_data) == 0:
        return {
            "rows": 0,
            "columns": [],
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "mode": 0.0,
            "range": 0.0,
            "allowed_values": {"min": -1.0, "max": 1.0, "valid": True},
            "value_range": [0.0, 0.0],
            "correlation": [0.0]
        }
    
    # Basic statistics
    mean_val = float(np.mean(audio_data))
    std_val = float(np.std(audio_data))
    variance_val = float(np.var(audio_data))
    min_val = float(np.min(audio_data))
    max_val = float(np.max(audio_data))
    median_val = float(np.median(audio_data))
    range_val = max_val - min_val
    value_range = [min_val, max_val]
    
    # Mode (most frequent value using histogram)
    hist, bin_edges = np.histogram(audio_data, bins=50)
    mode_idx = np.argmax(hist)
    mode_val = float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
    
    # Correlation (autocorrelation at lag 1)
    if len(audio_data) > 1:
        corr = float(np.corrcoef(audio_data[:-1], audio_data[1:])[0, 1])
    else:
        corr = 0.0
    
    # Allowed values check
    allowed_values = {
        "min": -1.0,
        "max": 1.0,
        "valid": bool(np.all((audio_data >= -1.0) & (audio_data <= 1.0)))
    }
    
    # Columns (feature names)
    columns = [
        "amplitude",
        "frequency",
        "energy",
        "zero_crossing_rate",
        "spectral_centroid"
    ]
    
    response = {
        "rows": len(audio_data),
        "columns": columns,
        "mean": mean_val,
        "std": std_val,
        "variance": variance_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "mode": mode_val,
        "range": range_val,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": [corr]
    }
    
    return response

@app.route('/your-api', methods=['POST'])
def process_audio():
    """Main API endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        audio_id = data.get('audio_id', 'unknown')
        audio_base64 = data.get('audio_base64')
        
        if not audio_base64:
            return jsonify({"error": "Missing audio_base64"}), 400
        
        print(f"Processing audio: {audio_id}")
        
        # Decode audio
        audio_data = decode_audio(audio_base64)
        
        if audio_data is None:
            return jsonify({"error": "Failed to decode audio"}), 400
        
        # Extract features
        result = extract_features(audio_data)
        
        # Return JSON response
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
