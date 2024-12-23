const WebSocket = require("ws");
const {startSpeedBroadcast} = require("./util/carManager");
const {maxSpeed} = process.env;
const {printToConsole} = require("./util/util");  // Import the file system module
const {currentDriveObject, updateDriveDataLog} = require("../../util/driveLogManager");

let detectionUnitData = undefined;

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
            handleClientMessage(ws, message, wss);
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
        detectionUnitData: detectionUnitData
    };
    ws.send(JSON.stringify(welcomeMessage));
}

// Function to broadcast a message to all clients
function broadcast(wss, data) {
    const message = JSON.stringify(data);
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// Function to handle messages from clients
function handleClientMessage(ws, message, wss) {
    try {
        const data = JSON.parse(message);
        const response = {
            type: null,
            data: data,
            event: null
        };

        const {type, msgData} = data;

        printToConsole("Received type: " + type)

        switch (type) {
            case "detection_feed":
                // Decode the image frame (base64 to buffer)
                if (!("face_tensors" in msgData) || !("frame" in msgData)) return;

                const {frame, face_tensors, face_landmark_tensors, commands} = msgData;

                if (typeof commands !== 'undefined')
                    detectionUnitData = commands

                // Broadcast detection data to all connected clients
                broadcast(wss, {
                    type: "detection_feed",
                    msgData: {
                        frame: frame,
                        ...(typeof face_tensors !== 'undefined' && { face_tensors }),
                        ...(typeof face_landmark_tensors !== 'undefined' && { face_landmark_tensors }),
                    }
                });
                break;

            case "accelerate":
                printToConsole(msgData);
                break;

            case "decelerate":
                printToConsole(msgData);
                break;

            case "cruise":
                printToConsole(msgData);
                break;

            case "alert":
                const {event} = data
                // TODO: here I should sound the message...

                // Example usage
                muteAllStreams();         // Mute all other streams
                playSound('/path/to/your_sound_file.wav');  // Play custom sound
                setTimeout(unmuteAllStreams, 3000); // Unmute after 3 seconds

                printToConsole(event);
                break;

            default:
                printToConsole(`unknown type: ${data}`)
                response.type = "unknown";

        }

        ws.send(JSON.stringify(response));
    } catch (error) {
        console.error("Invalid JSON received:", error);
    }

        try {
            // Parse the JSON data
            const data = JSON.parse(message);



            // Further processing can be done here, like passing the data to a frontend
        } catch (error) {
            console.error('Failed to process message:', error);
        }


}

// Export the initialization function and other WebSocket functions if needed
module.exports = {initWebSocket, broadcast};
