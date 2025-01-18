let gauge = null;
let lastFrameTime = null;
let samples = 1;
let fpsAvg = 0;

function createGauge(gaugeConf){
    if (!"tickList" in gaugeConf) {
        return 1;
    }
        if (!"redline" in gaugeConf) {
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

// Function to send the WebSocket message
function sendWebSocketMessage(type) {
    if (socket.readyState === WebSocket.OPEN) {
        const message = {
            type: type,      // Replace or parameterize as needed
            msgData: ""      // Add data if necessary
        };
        socket.send(JSON.stringify(message));
        console.log('Sent message:', message);
    } else {
        console.error('WebSocket is not open. Ready state:', socket.readyState);
    }
}

// Dynamically update speed gauge value
function updateSpeed(speed) {
    if (speed <= 0){
        gauge.value = 0;
        gauge.valueText = 0
        document.getElementById('speedValue').textContent = "0";
    }else {
        gauge.value = speed;
        gauge.valueText = speed
        document.getElementById('speedValue').textContent = speed;
    }
}

//const socket = new WebSocket("ws://192.168.0.233:5000");
// const socket = new WebSocket("ws://192.168.0.63:5000");
const socket = new WebSocket("ws://localhost:5000");

socket.onopen = () => {
    console.log("Connected to WebSocket server.");
};

socket.onmessage = (event) => {
    // Parse the JSON message received from the server
    const data = JSON.parse(event.data);
    const {type, msgData} = data;

    switch (type) {
        case "welcome":
            const { gaugeConf, systemData } = data;

            const systemDataHTML = `
                                <strong>CPU Model:</strong> ${systemData.cpuModel || "Unknown"}<br>
                                <strong>Memory:</strong><br>
                                  - Total: ${systemData.memory?.total || "N/A"}<br>
                                `;
            // Display the formatted systemData in the HTML element
            document.getElementById("detection-unit-data").innerHTML = systemDataHTML;

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
            const currentTime = performance.now(); // Get the current time in milliseconds
            if (lastFrameTime) {
                const fps = 1000 / (currentTime - lastFrameTime); // Calculate FPS
                document.getElementById("fpsDisplay").textContent = `FPS: ${fps.toFixed(2)}`;
                fpsAvg += fps
                document.getElementById("fpsDisplayAVG").textContent = `AVG. FPS: ${(fpsAvg / samples).toFixed(1)}`;
                samples += 1;
                if (fps < 25)
                    document.getElementById("fpsDisplay").style.color = "red";
                else
                    document.getElementById("fpsDisplay").style.color = "#00bf07";

            }
            lastFrameTime = currentTime; // Update the last frame time
            break;

        default:
            console.log("unknown type");
            break;
    }
};

socket.onclose = () => {
    console.log("Disconnected from WebSocket server.");
};

socket.onerror = (error) => {
    console.error("WebSocket error:", error);
};

// Add click event listener to the button
document.getElementById('confirm').addEventListener('click', function () {
    const messageType = 'manual_user_confirmation'; // Define the type as needed
    sendWebSocketMessage(messageType);
});



const wssn = new WebSocket('ws://192.168.0.252:8765/');  // Adjust host/port as needed

wssn.onopen = () => {
    console.log("Connected to the WebSocket server");
};

wssn.onmessage = (event) => {
    // event.data is the base64-encoded JPEG
    const base64Image = event.data;
    const imgElem = document.getElementById("video-frame");
    imgElem.src = "data:image/jpeg;base64," + base64Image;
};

wssn.onclose = () => {
    console.log("WebSocket connection closed");
};

const wssnn = new WebSocket('ws://192.168.0.63:8765/');  // Adjust host/port as needed

wssnn.onopen = () => {
    console.log("Connected to the WebSocket server");
};

wssnn.onmessage = (event) => {
    // event.data is the base64-encoded JPEG
    const base64Image = event.data;
    const imgElem = document.getElementById("videoFeed");
    imgElem.src = "data:image/jpeg;base64," + base64Image;
};

wssnn.onclose = () => {
    console.log("WebSocket connection closed");
};
