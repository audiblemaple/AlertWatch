// Speed Gauge
	var gauge = new RadialGauge({
		renderTo: 'canvas-id',
		width: 300,
		height: 300,
		units: "Km/h",
		minValue: 0,
		maxValue: 220,
		majorTicks: [
			"0",
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
		minorTicks: 2,
		strokeTicks: true,
		highlights: [
			{
				"from": 160,
				"to": 220,
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
		animationDuration: 1000,
		animationRule: "linear"
	}).draw();

	// Dynamically update speed gauge value
	function updateSpeed(speed) {
		gauge.value = speed;
		gauge.valueText = speed
		document.getElementById('speedValue').textContent = speed;
	}

	// Simulating speed updates
	let speed = 0;
	setInterval(() => {
		let selector = Math.random() > 0.5 ? 1 : -1; // Randomly select whether to increase or decrease the speed
		let speedChange = Math.floor(Math.random() * 30);

		// Update speed, ensuring it stays within 0 and 220
		speed = Math.max(0, Math.min(220, speed + selector * speedChange));

		updateSpeed(speed);
	}, 1500);

	// Video feed (Placeholder for network video feed)
    // TODO: add receiving and displaying of the video feed and postprocessing
	const videoElement = document.getElementById('videoFeed');
	navigator.mediaDevices.getUserMedia({ video: true })
		.then(stream => {
			videoElement.srcObject = stream;
		})
		.catch(error => {
			console.error("Error accessing camera: ", error);
		});
