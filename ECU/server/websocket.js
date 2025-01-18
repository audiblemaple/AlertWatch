/**
 * @file websocketServer.js
 * @description This file sets up and manages the WebSocket server for handling client communication and broadcasting data.
 * @author Lior Jigalo
 * @license MIT
 */

/** Extract environment variables */
const {maxSpeed} = process.env;

/** Import WebSocket library */
const WebSocket = require("ws");

/** Import sound utility functions */
const {playSound, askForUserConfirmation} = require("./util/sound");

/** Import general utility functions */
const {printToConsole, getSystemData, getRandomInt, delay} = require("./util/util");

/** Import car management functions */
const {startSpeedBroadcast, decelerateCar, accelerateCar} = require("./util/carManager");

/** Import drive log management functions */
const {currentDriveObject, updateDriveDataLog} = require("./util/driveLogManager");

/** Import global variables */
const {user_status, locks, sounds, carState} = require("./util/global");

/**
 * @typedef {object} MessageData
 * @property {string} type - Type of the message.
 * @property {object} msgData - Data associated with the message.
 */

/** @type {undefined | object} */
/** Holds data from the detection unit*/
let detectionUnitData = undefined;

/**
 * @function initWebSocket
 * @description Initializes the WebSocket server and sets up event listeners for client communication.
 * @param {object} server - The HTTP server instance to attach the WebSocket server.
 * @returns {WebSocket.Server} The WebSocket server instance.
 */
function initWebSocket(server) {
    const wss = new WebSocket.Server({server});

    // Event listener for new connections
    wss.on("connection", (ws) => {

        // Send a welcome message to the new client
        sendWelcomeMessage(ws);

        // Handle messages from the client
        ws.on("message", (message) => {
            handleClientMessage(ws, message, wss).then(r => printToConsole("Returned from alert message"));
        });

        // Handle client disconnection
        ws.on("close", () => {
           // console.log("WebSocket client disconnected");
        });
    });

    // Start broadcasting speed updates using the WebSocket server instance
    startSpeedBroadcast(wss);
    return wss;
}

/**
 * @function sendWelcomeMessage
 * @description Sends a welcome message to the connected client, including system data and gauge configuration.
 * @param {WebSocket} ws - The WebSocket instance of the connected client.
 */
function sendWelcomeMessage(ws) {
    let tickList = [];
    for (let i = 0; i <= maxSpeed; i += 20)
        tickList.push(i);

    const systemData = getSystemData();

    const welcomeMessage = {
        type: "welcome",
        gaugeConf: {
            tickList: tickList,
            redline: {
                minVal: 160,
                maxVal: 220
            },
        },
        detectionUnitData: detectionUnitData,
        systemData: systemData
    };

    ws.send(JSON.stringify(welcomeMessage));
}

/**
 * @function broadcast
 * @description Sends a message to all connected WebSocket clients.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 * @param {object} data - The data to broadcast.
 */
function broadcast(wss, data) {
    const message = JSON.stringify(data);
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN)
            client.send(message);
    });
}

/**
 * @function handleClientMessage
 * @description Handles incoming messages from WebSocket clients.
 * @param {WebSocket} ws - The WebSocket instance of the client sending the message.
 * @param {string} message - The received message in JSON format.
 * @param {WebSocket.Server} wss - The WebSocket server instance.
 */
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

            case "manual_user_confirmation":
            case "accelerate":
                if (getRandomInt(10) > 5) accelerateCar();
                else decelerateCar();
                break;

            case "alert": {
                // Check if alert messages are locked (simple semaphore logic)
                if (locks.alert_lock) {
                    printToConsole("alert messages are locked");
                    return;
                }
                locks.alert_lock = true;

                console.log(msgData);

                switch (msgData) {
                    case "low_average_ear":
                        await playSound(sounds.takeABreak);
                        break;

                    case "high_blink_rate":
                        // not sure if this will be used yet
                        break;

                    case "Prolonged_eye_closure":
                        // If we’ve triggered a “medium alert” before, play attentionTest
                        // and loop until user responds or alert is escalated
                        if (currentDriveObject.medium_alert_num > 0) {
                            await playSound(sounds.attentionTest);

                            try {
                                // Wait for user confirmation in a loop
                                while (true) {
                                    const isConfirmedAlert = await askForUserConfirmation();

                                    if (isConfirmedAlert === user_status.userResponded) {
                                        // User confirmed alert
                                        printToConsole("User has confirmed being alert.");
                                        await playSound(sounds.gotIt);
                                        currentDriveObject.consecutive_alert_num = 0;
                                        break;
                                    } else if (isConfirmedAlert === user_status.noResponse) {
                                        // User did not confirm → increment counter
                                        currentDriveObject.consecutive_alert_num += 1;
                                        if (currentDriveObject.consecutive_alert_num >= 1)
                                            break;

                                        printToConsole("User did not confirm being alert.");
                                        await playSound(sounds.noResponse);
                                    } else if (isConfirmedAlert === user_status.failedToParse)
                                        // Retry if parse failed
                                        await playSound(sounds.failedToParse);
                                }

                                // If user never responded, escalate: play decelerateWarning
                                if (currentDriveObject.consecutive_alert_num >= 1) {
                                    printToConsole("debug 5");
                                    currentDriveObject.consecutive_alert_num = 0;

                                    await playSound(sounds.failedToParse);
                                    const isConfirmedAlert = await askForUserConfirmation();

                                    if (isConfirmedAlert === user_status.userResponded) {
                                        await playSound(sounds.gotIt);
                                        currentDriveObject.consecutive_alert_num = 0;
                                    } else if (isConfirmedAlert === user_status.failedToParse) {
                                        // Should ideally loop until response is valid, ill do that later...
                                        await playSound(sounds.failedToParse);
                                    } else {
                                        // TODO: find a way to interrupt this process (currently, manual confirmation should suffice)
                                        await playSound(sounds.decelerating);
                                        decelerateCar();
                                        const count = 60;
                                        let counter = 0;
                                        while (carState.decelerating === true || counter <= count){
                                            await playSound(sounds.beep);
                                            await delay(500);
                                            counter++
                                        }
                                    }
                                }

                            } catch (err) {
                                // In case anything goes wrong
                                console.error("Error:", err.message);
                                locks.alert_lock = true;
                            }
                            break; // End of "alert" logic when medium_alert_num > 0
                        }
                        // If this is the first time medium_alert_num = 0, just play "takeABreak"
                        // and increment the alert count
                        else {
                            await playSound(sounds.takeABreak);
                            currentDriveObject.medium_alert_num += 1;
                        }
                        break;

                    default:
                        return
                }
                break; // End of case "alert"
            }

            default:
                printToConsole(`unknown type: ${data}`)
                response.type = "unknown";
                locks.alert_lock = false;

        }
        locks.alert_lock = false;

        ws.send(JSON.stringify(response));
    } catch (error) {
        console.error("Invalid JSON received:", error);
    }
}

/** Export WebSocket functions */
module.exports = {initWebSocket, broadcast};