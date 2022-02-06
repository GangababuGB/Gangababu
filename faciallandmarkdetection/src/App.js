 // 1. Install dependencies DONE
// 2. Import dependencies DONE
// 3. Setup webcam and canvas 
// 4. Define references to those 
// 5. Load posenet 
// 6. Detect function 
// 7. Drawing utilities from tensorflow 
// 8. Draw functions 

import React, { useRef, useEffect } from "react";
import './App.css';
// import * as tf from "@tensorflow/tfjs";
// import * as facemesh from "@tensorflow-models/facemesh";
import * as facemesh from "@tensorflow-models/face-landmarks-detection";
import Webcam  from "react-webcam";
import { drawMesh } from "./utilities";

function App() {
  // setup references
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Load posenet
  const runFacemesh = async () => {
    //old model
    //     const net = await facemesh.load({
    //       inputResolution: { width: 640, height: 480 },
    //       scale: 0.8,
    // });
    // new model
    const net = await facemesh.load(facemesh.SupportedPackages.mediapipeFacemesh);
    setInterval(() => {
      detect(net);
    }, 10);
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      // OLD MODEL
      //       const face = await net.estimateFaces(video);
      // NEW MODEL
      const face = await net.estimateFaces({input:video});
      console.log(face);

      // Get canvas context
      const ctx = canvasRef.current.getContext("2d");
      requestAnimationFrame(()=>{drawMesh(face, ctx)});
    }
  };

  useEffect(()=>{runFacemesh()}, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam  
        ref = {webcamRef} 
        style={{
          position:"absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }} 
      />
      <canvas
      ref={canvasRef} 
      style={{
          position:"absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }} 
      />
      </header>
    </div>
  );
}

export default App;
