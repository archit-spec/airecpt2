const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

let socket;
let isConnecting = false;

function connectWebSocket() {
    if (isConnecting) return;
    isConnecting = true;

    socket = new WebSocket('ws://localhost:8000/ws');

    socket.onopen = function() {
        console.log('WebSocket connected');
        isConnecting = false;
        addMessage("Welcome to Dr. Adrin's office. Are you having an emergency or would you like to leave a message?", 'ai');
    };

    socket.onmessage = function(event) {
        addMessage(event.data, 'ai');
    };

    socket.onclose = function(event) {
        console.log('WebSocket closed. Attempting to reconnect...');
        isConnecting = false;
        setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        isConnecting = false;
    };
}

connectWebSocket();

chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const message = userInput.value;
    if (message.trim() === '') return;

    addMessage(message, 'user');
    sendMessage(message);
    userInput.value = '';
});

function sendMessage(message) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(message);
    } else {
        console.log('WebSocket is not open. Attempting to reconnect...');
        connectWebSocket();
        setTimeout(() => sendMessage(message), 1000);
    }
}

function addMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}