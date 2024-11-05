const WebSocket = require("ws");
let speed = 0;

// Function to generate a random speed update
function updateSpeed() {
  const selector = Math.random() > 0.5 ? 1 : -1; // Randomly select whether to increase or decrease the speed
  const speedChange = Math.floor(Math.random() * 30);
  speed = Math.max(0, Math.min(220, speed + selector * speedChange));
  return speed;
}

// Function to start broadcasting speed updates
function startSpeedBroadcast(wss) {
  setInterval(() => {
    const newSpeed = updateSpeed();
    broadcastSpeed(wss, newSpeed);
  }, 1500);
}

// Function to broadcast speed to all connected WebSocket clients
function broadcastSpeed(wss, speed) {
  const message = JSON.stringify({
    type: "speed",
    msgData: speed
  });

  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

module.exports = { startSpeedBroadcast };
