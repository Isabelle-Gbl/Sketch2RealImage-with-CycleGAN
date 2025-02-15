from interaction.helpers import *




# Setup-Funktion einmal beim Start der App ausführen
# Define directories for image storage
cwd = os.getcwd()

app = Flask(__name__,template_folder="interaction/templates")
socketio = SocketIO(app, cors_allowed_origins="*")








@app.route('/get_models', methods=['GET'])
def get_models():
    """Returns available model names as JSON."""
    models = get_available_models()
    return jsonify(models)

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    """Receives an image, processes it, and runs the model transformation."""
    data = request.get_json()
    image_data, model_name = data.get('imageData'), data.get('model_selction')
    
    if not model_name:
        return jsonify({"error": "No valid model selected."}), 400
    
    image_bytes = io.BytesIO(base64.b64decode(image_data.split(",")[1]))
    img = ensure_white_background(Image.open(image_bytes))
    
    try:
        process_and_save_image(img)
        run_model(model_name)
        
        return jsonify({
            "input_image": f"/get_image/{os.path.basename(input_image_path_app)}",
            "output_image": f"/get_image/{os.path.basename(upload_image_path_app)}"
        })

    except ValueError as e:
       
        return jsonify({"error": str(e)}), 400



@app.route('/get_image/<image_name>')
def get_image(image_name):
    """Serves images from different directories."""
    possible_paths = [
        os.path.join(cwd, 'interaction', 'images','input', image_name),
        os.path.join(cwd, 'interaction', 'images','output', image_name),
        os.path.join(cwd, 'interaction', 'images','upload', image_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return send_file(path, mimetype='image/png')
    return jsonify({"error": "Image not found."}), 404






# WebSocket Events
connected_clients = set()

@socketio.on('connect')
def handle_connect():
    connected_clients.add(request.sid)
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    connected_clients.discard(request.sid)
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_new_path')
def handle_start_new_path(data):
    emit('new_path', data, broadcast=True, include_self=False)

@socketio.on('draw_data')
def handle_draw_data(data):
    emit('update_canvas', data, broadcast=True, include_self=False)

@socketio.on('reset_canvas')
def handle_reset_canvas():
    emit('clear_canvas', broadcast=True, include_self=False)



if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

