const WebSocket = require("ws");
const {startSpeedBroadcast} = require("./util/carManager");
const {maxSpeed} = process.env;

// Function to initialize the WebSocket server
function initWebSocket(server) {
    const wss = new WebSocket.Server({server});

    // Event listener for new connections
    wss.on("connection", (ws) => {
        console.log("New WebSocket client connected");

        // Send a welcome message to the new client
        sendWelcomeMessage(ws);

        // Handle messages from the client
        ws.on("message", (message) => {
            handleClientMessage(ws, message);
        });

        // Handle client disconnection
        ws.on("close", () => {
            console.log("WebSocket client disconnected");
        });
    });

    // Start broadcasting speed updates using the WebSocket server instance
    startSpeedBroadcast(wss);
    return wss;
}

// Function to send a welcome message
function sendWelcomeMessage(ws) {
    let tickList = [];
    for (let i = 0; i <= maxSpeed; i += 20) {
        tickList.push(i);
    }
    const welcomeMessage = {
        type: "welcome",
        gaugeConf: {
            tickList: tickList,
            redline: {
                minVal: 160,
                maxVal: 220
            },
        },
    };
    ws.send(JSON.stringify(welcomeMessage));
}

// Function to broadcast a message to all clients
function broadcast(wss, data) {
    const message = JSON.stringify(data);
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN)
            client.send(message);
    });
}

module.exports = {initWebSocket, broadcast};

// Function to handle messages from clients
function handleClientMessage(ws, message) {
    try {
        const data = JSON.parse(message);
        console.log("Received JSON from client:", data);

        const response = {
            type: null,
            data: data,
        };

        const {type, msgData} = message;

        switch (type) {
            case "accelerate":
                console.log(msgData);
                break;

            case "decelerate":
                console.log(msgData);
                break;

            case "cruise":
                console.log(msgData);
                break;

            default:
                response.type = "unknown";

        }

        ws.send(JSON.stringify(response));
    } catch (error) {
        console.error("Invalid JSON received:", error);
    }
}

// Export the initialization function and other WebSocket functions if needed
module.exports = {initWebSocket};
