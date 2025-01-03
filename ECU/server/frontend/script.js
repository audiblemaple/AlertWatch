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

//const socket = new WebSocket("ws://192.168.0.233:5000");
const socket = new WebSocket("ws://192.168.0.63:5000");

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
<!--                                              - Used: ${systemData.memory?.used || "N/A"}<br>-->
<!--                                              - Free: ${systemData.memory?.free || "N/A"}<br>-->
<!--                                              - Shared: ${systemData.memory?.shared || "N/A"}<br>-->
<!--                                              - Buffer/Cache: ${systemData.memory?.bufferCache || "N/A"}<br>-->
<!--                                               - Available: ${systemData.memory?.available || "N/A"}<br>-->
<!--                                            <strong>Swap:</strong><br>-->
<!--                                              - Total: ${systemData.memory?.swap?.total || "N/A"}<br>-->
<!--                                              - Used: ${systemData.memory?.swap?.used || "N/A"}<br>-->
<!--                                              - Free: ${systemData.memory?.swap?.free || "N/A"}<br>-->
                                            <strong>Hailo Information:</strong><br>
<!--                                              - Executing Device: ${systemData.hailoInfo?.ExecutingDevice || "N/A"}<br>-->
                                              - Control Protocol Version: ${systemData.hailoInfo?.ControlProtocolVersion || "N/A"}<br>
                                              - Firmware Version: ${systemData.hailoInfo?.FirmwareVersion || "N/A"}<br>
<!--                                              - Logger Version: ${systemData.hailoInfo?.LoggerVersion || "N/A"}<br>-->
                                              - Board Name: ${systemData.hailoInfo?.BoardName || "N/A"}<br>
                                              - Device Architecture: ${systemData.hailoInfo?.DeviceArchitecture || "N/A"}<br>
<!--                                              - Serial Number: ${systemData.hailoInfo?.SerialNumber || "N/A"}<br>-->
<!--                                              - Part Number: ${systemData.hailoInfo?.PartNumber || "N/A"}<br>-->
<!--                                              - Product Name: ${systemData.hailoInfo?.ProductName || "N/A"}<br>-->
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