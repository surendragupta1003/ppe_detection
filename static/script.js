document.addEventListener("DOMContentLoaded", function() {
    const cameraSelect = document.getElementById("cameraSelect");
    const videoFeed = document.getElementById("videoFeed");
    const motionLog = document.getElementById("motionLog");

    function updateVideoStream() {
        const selectedCamera = cameraSelect.value;
        videoFeed.src = `/video_feed?camera=${selectedCamera}`;
    }

    async function fetchMotionLog() {
        const response = await fetch("/motion_log");
        const logs = await response.json();
        motionLog.innerHTML = logs.map(log => `<li>${log[1]} - ${log[2]}</li>`).join("");
    }

    cameraSelect.addEventListener("change", updateVideoStream);
    updateVideoStream();
    setInterval(fetchMotionLog, 5000);
});
