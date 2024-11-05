const WebSocket = require("ws");
const {startSpeedBroadcast} = require("./util/carManager");

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
    const welcomeMessage = {
        type: "welcome",
        gaugeConf: {
            tickList: [ "0",
                    "20",
                    "40",
                    "60",
                    "80",
                    "100",
                    "120",
                    "140",
                    "160",
                    "180",
                    "200",
                    "220"
            ],
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
            case "speed":
                console.log(msgData);
                break;

            case "TODO":
                console.log();
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
