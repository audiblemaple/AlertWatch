let gauge = null;
let lastFrameTime = null;
let samples = 1;
let fpsAvg = 0;

function createGauge(gaugeConf) {
    if (!("tickList" in gaugeConf)) {
        return 1;
    }
    if (!("redline" in gaugeConf)) {
        return 1;
    }

    // Speed Gauge
    gauge = new RadialGauge({
        renderTo: 'canvas-id',
        width: 300,
        height: 300,
        units: "Km/h",
        minValue: 0,
        maxValue: 220,
        majorTicks: gaugeConf.tickList,
        minorTicks: 2,
        strokeTicks: true,
        highlights: [
            {
                "from": gaugeConf.redline.minVal,
                "to": gaugeConf.redline.maxVal,
                "color": "rgba(200, 50, 50, .75)"
            }
        ],
        colorPlate: "#fff",
        borderShadowWidth: 0,
        borders: false,
        needleType: "arrow",
        needleWidth: 2,
        needleCircleSize: 7,
        needleCircleOuter: true,
        needleCircleInner: false,
        animationDuration: 970,
        animationRule: "linear"
    }).draw();
}

// Dynamically update speed gauge value
function updateSpeed(speed) {
    if (speed <= 0) {
        gauge.value = 0;
        gauge.valueText = 0;
        document.getElementById('speedValue').textContent = "0";
    } else {
        gauge.value = speed;
        gauge.valueText = speed;
        document.getElementById('speedValue').textContent = speed;
    }
}

// -------------------------------------------
// Main Socket (was: const socket = new WebSocket("ws://localhost:5000"); )
// -------------------------------------------

let socket = null;

function connectMainSocket() {
    socket = new WebSocket("ws://localhost:5000");
    // socket = new WebSocket("ws://192.168.0.63:5000");

    socket.onopen = () => {
        console.log("Connected to WebSocket server (main).");
    };

    socket.onmessage = (event) => {
        // Parse the JSON message received from the server
        const data = JSON.parse(event.data);
        const {type, msgData} = data;

        switch (type) {
            case "welcome":
                const {gaugeConf, systemData} = data;
                const systemDataHTML = `
                      <strong>CPU Model:</strong> ${systemData.cpuModel || "Unknown"}<br>
                      <strong>Cores:</strong> ${systemData.cpuCores || "Unknown"}<br><br>
                      <strong>Memory:</strong><br>
                      - Total: ${systemData.memory?.total || "N/A"}<br><br>
                      <strong>OS Info:</strong><br>
                      - Platform: ${systemData.osInfo.platform || "Unknown"}<br>
                      - Release: ${systemData.osInfo.release || "Unknown"}<br>
                      - Architecture: ${systemData.osInfo.arch || "Unknown"}<br>
                  `;
                document.getElementById("ECU-unit-data").innerHTML = systemDataHTML;
                createGauge(gaugeConf);
                break;

            case "speed":
                updateSpeed(msgData);
                break;

            case "detection_feed":
                // Extract the frame and update the image
                const {frame} = msgData;
                const videoElement = document.getElementById('videoFeed');
                videoElement.src = `data:image/jpeg;base64,${frame}`;

                // Calculate and display FPS
                const currentTime = performance.now();
                if (lastFrameTime) {
                    const fps = 1000 / (currentTime - lastFrameTime);
                    document.getElementById("fpsDisplay").textContent = `FPS: ${fps.toFixed(2)}`;
                    fpsAvg += fps;
                    document.getElementById("fpsDisplayAVG").textContent = `AVG. FPS: ${(fpsAvg / samples).toFixed(1)}`;
                    samples++;
                    if (fps < 25)
                        document.getElementById("fpsDisplay").style.color = "red";
                    else
                        document.getElementById("fpsDisplay").style.color = "#00bf07";
                }
                lastFrameTime = currentTime;
                break;

            default:
                console.log("Unknown message type");
                break;
        }
    };

    socket.onclose = () => {
        console.log("Disconnected from WebSocket server (main). Reconnecting in 3 seconds...");
        setTimeout(connectMainSocket, 3000); // Attempt to reconnect after 3s
    };

    socket.onerror = (error) => {
        console.error("WebSocket error (main):", error);
        // We can optionally close to trigger onclose and reconnect
        // socket.close();
    };
}

// Call once on page load
connectMainSocket();

// Send function
function sendWebSocketMessage(type) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        const message = {type, msgData: ""};
        socket.send(JSON.stringify(message));
        console.log('Sent message:', message);
    } else {
        console.error('Main WebSocket is not open. Ready state:', socket?.readyState);
    }
}

// Add click event to the button
document.getElementById('confirm').addEventListener('click', function () {
    const messageType = 'manual_user_confirmation';
    sendWebSocketMessage(messageType);
});


// -------------------------------------------
// videoFeedWebSocket Socket (was: const videoFeedWebSocket = new WebSocket('ws://192.168.0.252:8765/');)
// -------------------------------------------

let videoFeedWebSocket = null;

function connectWssn() {
    // Create the WebSocket
    // videoFeedWebSocket = new WebSocket('ws://192.168.0.63:8765/');
    videoFeedWebSocket = new WebSocket('ws://localhost:8765/');

    videoFeedWebSocket.onopen = () => {
        console.log("Connected to videoFeedWebSocket server (192.168.0.63:8765)");
    };

    videoFeedWebSocket.onmessage = (event) => {
        // Log the raw string

        let data;
        try {
            // Attempt to parse JSON
            data = JSON.parse(event.data);
        } catch (e) {
            // If it fails, we assume it's a base64 image
            data = null;
        }

        // If we successfully parsed a JSON object with "type"
        if (data && data.type === "welcome") {
            // We have system data

            // Update your UI with system information
            document.getElementById("detection-unit-data").innerHTML = `
                <strong>Processor:</strong> ${data.systemData.processor || "Unknown"}<br><br>
                <strong>System:</strong> ${data.systemData.platform || "Unknown"}<br>
                <strong>Release:</strong> ${data.systemData.platform_release || "Unknown"}<br><br>
                <strong>Architecture:</strong> ${data.systemData.architecture || "Unknown"}<br><br>
            `;
        } else {
            // Otherwise, treat the event data as a base64-encoded JPEG
            const base64Image = event.data;
            const imgElem = document.getElementById("video-frame");
            imgElem.src = "data:image/jpeg;base64," + base64Image;
        }
    };

    videoFeedWebSocket.onclose = () => {
        console.log("Disconnected from videoFeedWebSocket server. Reconnecting in 3 seconds...");
        setTimeout(connectWssn, 3000);
    };

    videoFeedWebSocket.onerror = (error) => {
        console.error("WebSocket error (videoFeedWebSocket):", error);
        // videoFeedWebSocket.close(); // Optionally close to trigger reconnect
    };
}


// Call once on page load
// connectWssn();


// let gauge = null;
// let lastFrameTime = null;
// let samples = 1;
// let fpsAvg = 0;
//
// function createGauge(gaugeConf){
//     if (!"tickList" in gaugeConf) {
//         return 1;
//     }
//         if (!"redline" in gaugeConf) {
//         return 1;
//     }
//
//     // Speed Gauge
//     gauge = new RadialGauge({
//         renderTo: 'canvas-id',
//         width: 300,
//         height: 300,
//         units: "Km/h",
//         minValue: 0,
//         maxValue: 220,
//         majorTicks: gaugeConf.tickList,
//         minorTicks: 2,
//         strokeTicks: true,
//         highlights: [
//             {
//                 "from": gaugeConf.redline.minVal,
//                 "to": gaugeConf.redline.maxVal,
//                 "color": "rgba(200, 50, 50, .75)"
//             }
//         ],
//         colorPlate: "#fff",
//         borderShadowWidth: 0,
//         borders: false,
//         needleType: "arrow",
//         needleWidth: 2,
//         needleCircleSize: 7,
//         needleCircleOuter: true,
//         needleCircleInner: false,
//         animationDuration: 970,
//         animationRule: "linear"
//     }).draw();
// }
//
// // Function to send the WebSocket message
// function sendWebSocketMessage(type) {
//     if (socket.readyState === WebSocket.OPEN) {
//         const message = {
//             type: type,      // Replace or parameterize as needed
//             msgData: ""      // Add data if necessary
//         };
//         socket.send(JSON.stringify(message));
//         console.log('Sent message:', message);
//     } else {
//         console.error('WebSocket is not open. Ready state:', socket.readyState);
//     }
// }
//
// // Dynamically update speed gauge value
// function updateSpeed(speed) {
//     if (speed <= 0){
//         gauge.value = 0;
//         gauge.valueText = 0
//         document.getElementById('speedValue').textContent = "0";
//     }else {
//         gauge.value = speed;
//         gauge.valueText = speed
//         document.getElementById('speedValue').textContent = speed;
//     }
// }
//
// //const socket = new WebSocket("ws://192.168.0.233:5000");
//  const socket = new WebSocket("ws://192.168.0.63:5000");
// //const socket = new WebSocket("ws://localhost:5000");
//
// socket.onopen = () => {
//     console.log("Connected to WebSocket server.");
// };
//
// socket.onmessage = (event) => {
//     // Parse the JSON message received from the server
//     const data = JSON.parse(event.data);
//     const {type, msgData} = data;
//
//     switch (type) {
//         case "welcome":
//             const { gaugeConf, systemData } = data;
//
//             const systemDataHTML = `
//                                 <strong>CPU Model:</strong> ${systemData.cpuModel || "Unknown"}<br>
//                                 <strong>Memory:</strong><br>
//                                   - Total: ${systemData.memory?.total || "N/A"}<br>
//                                 `;
//             // Display the formatted systemData in the HTML element
//             document.getElementById("detection-unit-data").innerHTML = systemDataHTML;
//
//             createGauge(gaugeConf);
//             break;
//
//         case "speed":
//             updateSpeed(msgData);
//             break;
//
//         case "detection_feed":
//             // Extract the frame and update the image
//             const {frame} = msgData;
//             const videoElement = document.getElementById('videoFeed');
//             videoElement.src = `data:image/jpeg;base64,${frame}`;
//
//             // Calculate and display FPS
//             const currentTime = performance.now(); // Get the current time in milliseconds
//             if (lastFrameTime) {
//                 const fps = 1000 / (currentTime - lastFrameTime); // Calculate FPS
//                 document.getElementById("fpsDisplay").textContent = `FPS: ${fps.toFixed(2)}`;
//                 fpsAvg += fps
//                 document.getElementById("fpsDisplayAVG").textContent = `AVG. FPS: ${(fpsAvg / samples).toFixed(1)}`;
//                 samples += 1;
//                 if (fps < 25)
//                     document.getElementById("fpsDisplay").style.color = "red";
//                 else
//                     document.getElementById("fpsDisplay").style.color = "#00bf07";
//
//             }
//             lastFrameTime = currentTime; // Update the last frame time
//             break;
//
//         default:
//             console.log("unknown type");
//             break;
//     }
// };
//
// socket.onclose = () => {
//     console.log("Disconnected from WebSocket server.");
// };
//
// socket.onerror = (error) => {
//     console.error("WebSocket error:", error);
// };
//
// // Add click event listener to the button
// document.getElementById('confirm').addEventListener('click', function () {
//     const messageType = 'manual_user_confirmation'; // Define the type as needed
//     sendWebSocketMessage(messageType);
// });
//
//
//
// const videoFeedWebSocket = new WebSocket('ws://192.168.0.252:8765/');  // Adjust host/port as needed
//
// videoFeedWebSocket.onopen = () => {
//     console.log("Connected to the WebSocket server");
// };
//
// videoFeedWebSocket.onmessage = (event) => {
//     // event.data is the base64-encoded JPEG
//     const base64Image = event.data;
//     const imgElem = document.getElementById("video-frame");
//     imgElem.src = "data:image/jpeg;base64," + base64Image;
// };
//
// videoFeedWebSocket.onclose = () => {
//     console.log("WebSocket connection closed");
// };
//
// const videoFeedWebsocket = new WebSocket('ws://192.168.0.63:8765/');  // Adjust host/port as needed
//
// videoFeedWebsocket.onopen = () => {
//     console.log("Connected to the WebSocket server");
// };
//
// videoFeedWebsocket.onmessage = (event) => {
//     // event.data is the base64-encoded JPEG
//     const base64Image = event.data;
//     const imgElem = document.getElementById("videoFeed");
//     imgElem.src = "data:image/jpeg;base64," + base64Image;
// };
//
// videoFeedWebsocket.onclose = () => {
//     console.log("WebSocket connection closed");
// };
