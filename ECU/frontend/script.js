var gauge = null;
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
        animationDuration: 950,
        animationRule: "linear"
    }).draw();
}

// Dynamically update speed gauge value
function updateSpeed(speed) {
    console.log(speed);
    gauge.value = speed;
    gauge.valueText = speed
    document.getElementById('speedValue').textContent = speed;
}

// Video feed (Placeholder for network video feed)
// TODO: add receiving and displaying of the video feed and postprocessing
const videoElement = document.getElementById('videoFeed');
navigator.mediaDevices.getUserMedia({video: true})
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(error => {
        console.error("Error accessing camera: ", error);
    });

// const socket = new WebSocket("ws://192.168.0.233:5000");
const socket = new WebSocket("ws://192.168.0.64:5000");

socket.onopen = () => {
    console.log("Connected to WebSocket server.");
};

socket.onmessage = (event) => {
    // Parse the JSON message received from the server
    const data = JSON.parse(event.data);
    const {type, msgData} = data;
    console.log(data);

    switch (type) {
		case "welcome":
            const {gaugeConf} = data;
            createGauge(gaugeConf);
            console.warn("welcome messages are just for debugging");
            break;

        case "speed":
            updateSpeed(msgData);
            break;

        default:
            console.log("unknown type");
            break;
    }

    console.log(data);
};

socket.onclose = () => {
    console.log("Disconnected from WebSocket server.");
};

socket.onerror = (error) => {
    console.error("WebSocket error:", error);
};