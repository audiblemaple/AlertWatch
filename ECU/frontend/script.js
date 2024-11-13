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

// Dynamically update speed gauge value
function updateSpeed(speed) {
    gauge.value = speed;
    gauge.valueText = speed
    document.getElementById('speedValue').textContent = speed;
}

const socket = new WebSocket("ws://192.168.0.233:5000");

socket.onopen = () => {
    console.log("Connected to WebSocket server.");
};

socket.onmessage = (event) => {
    // Parse the JSON message received from the server
    const data = JSON.parse(event.data);
    const {type, msgData} = data;

    switch (type) {
		case "welcome":
            const {gaugeConf, detectionUnitData} = data;
            console.log(detectionUnitData);
            document.getElementById("detection-unit-data").textContent = detectionUnitData;

            createGauge(gaugeConf);
            console.warn("welcome messages are just for debugging");
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