const path = require("path");
const {maxSpeed} = process.env;
const WebSocket = require("ws");
const {playSound, transcribeWithWhisper} = require("./util/sound");
const {printToConsole} = require("./util/util");  // Import the file system module
const {startSpeedBroadcast} = require("./util/carManager");
const {currentDriveObject, updateDriveDataLog} = require("./util/driveLogManager");

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
async function handleClientMessage(ws, message, wss) {
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

//                if (currentDriveObject.high_alert_num > 0){
//                    const assetDir = path.join(__dirname, "assets/sounds");
//                    const filePath = path.join(assetDir, "break2.wav");
//                    playSound(filePath);
//                    break;
//                }
//                else if (currentDriveObject.medium_alert_num > 0) {
                if (currentDriveObject.medium_alert_num > 0) {
                    let wavFilePath = path.join(__dirname, 'assets/sounds', 'attentionTest.wav');
                    playSound(wavFilePath);

                    wavFilePath = path.join(__dirname, 'assets/sounds', 'attentionTest.wav');
                    try {
                        const wavFilePath = path.join(__dirname, "assets/sounds", "attentionTest.wav");

                        // 1) Get raw output from whisper-cli
                        const rawOutput = await transcribeWithWhisper(wavFilePath);
                        console.log("Raw output:\n", rawOutput);

                        // 2) Optionally parse out timestamps
                        const linesWithoutTimestamps = parseWhisperOutput(rawOutput);
                        console.log("\nTranscription lines (no timestamps):\n", linesWithoutTimestamps);
                    } catch (err) {
                        console.error("Error:", err.message);
                    }

                    currentDriveObject.high_alert_num += 1;
                    break;
                } else {
                    const assetDir = path.join(__dirname, "assets/sounds");
                    const filePath = path.join(assetDir, "break2.wav");
                    playSound(filePath);
                    currentDriveObject.medium_alert_num += 1;
                }

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
