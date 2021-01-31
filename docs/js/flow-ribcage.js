/* 
Web bluetooth api script for BreathZpot sensors. 
Adapted from the following source: https://googlechrome.github.io/samples/web-bluetooth/notifications-async-await.html

Copyright 2019 Eigil Aandahl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

var flowRibcageCharacteristic;
var ribcageValues = [];
var airflowValues = [];
var recentAirflow = [];
var maxRibVal = 0;
var minRibVal = 4096;

var minAirVal = 1000;
var maxAirVal = -1000;

var ribcageCanvas = document.querySelector('#ribcageChart');
var airflowCanvas = document.querySelector('#airflowChart');

var flowAbdomenCharacteristic;
var abdomenValues = [];
var maxAbVal = 0;
var minAbVal = 4096;
var abdomenCanvas = document.querySelector('#abdomenChart');

async function onFlowRibcageButtonClick() {

    let serviceUuid = 0xffb0; // BreathZpot breathing service
    serviceUuid = parseInt(serviceUuid);

    let characteristicUuid = 0xffb3; // BreathZpot breathing characteristic
    characteristicUuid = parseInt(characteristicUuid);

    try {
        console.log('Requesting Bluetooth Device...');
        const device = await navigator.bluetooth.requestDevice({
            filters: [{services: [serviceUuid]}]});

        console.log('Connecting to GATT Server...');
        const server = await device.gatt.connect();

        console.log('Getting Service...');
        const service = await server.getPrimaryService(serviceUuid);

        console.log('Getting Characteristic...');
        flowRibcageCharacteristic = await service.getCharacteristic(characteristicUuid);

        await flowRibcageCharacteristic.startNotifications();

        console.log('> Notifications started');
        flowRibcageCharacteristic.addEventListener('characteristicvaluechanged',
            handleFlowRibcageNotifications);

    
    } catch(error) {
        console.log('Argh! ' + error);
    }

}

async function onFlowAbdomenButtonClick() {

    let serviceUuid = 0xffb0; // BreathZpot breathing service
    serviceUuid = parseInt(serviceUuid);

    let characteristicUuid = 0xffb3; // BreathZpot breathing characteristic
    characteristicUuid = parseInt(characteristicUuid);

    try {
        console.log('Requesting Bluetooth Device...');
        const device = await navigator.bluetooth.requestDevice({
            filters: [{services: [serviceUuid]}]});

        console.log('Connecting to GATT Server...');
        const server = await device.gatt.connect();

        console.log('Getting Service...');
        const service = await server.getPrimaryService(serviceUuid);

        console.log('Getting Characteristic...');
        flowAbdomenCharacteristic = await service.getCharacteristic(characteristicUuid);

        await flowAbdomenCharacteristic.startNotifications();

        console.log('> Notifications started');
        flowAbdomenCharacteristic.addEventListener('characteristicvaluechanged',
            handleFlowAbdomenNotifications);

    
    } catch(error) {
        console.log('Argh! ' + error);
    }

}

async function onStopFlowRibcageClick() {
  if (flowRibcageCharacteristic) {
    try {
      await flowRibcageCharacteristic.stopNotifications();
      console.log('> Notifications stopped');
      flowRibcageCharacteristic.removeEventListener('characteristicvaluechanged',
          handleFlowRibcageNotifications);
    } catch(error) {
      console.log('Argh! ' + error);
    }
  }
}

async function onStopFlowAbdomenClick() {
  if (flowAbdomenCharacteristic) {
    try {
      await flowAbdomenCharacteristic.stopNotifications();
      console.log('> Notifications stopped');
      flowAbdomenCharacteristic.removeEventListener('characteristicvaluechanged',
          handleFlowAbdomenNotifications);
    } catch(error) {
      console.log('Argh! ' + error);
    }
  }
}

function handleFlowRibcageNotifications(event) {
    let value = event.target.value;
    let id = event.target.service.device.id;
    let int16View = new Int16Array(value.buffer);
    let timestamp = new Date().getTime();
    // TextDecoder to process raw data bytes.
    for (let i = 0; i < 7; i++) {

        let v = int16View[i];

        if (v > maxRibVal) {
            maxRibVal = v;
        }
        if (v < minRibVal) {
            minRibVal = v;
        }

        ribcageValues.push(int16View[i]);
    }
    ribcageText.innerHTML = "Ribcage movement: " + int16View[0].toString() + " mV";
    
    // let minRibVal = Math.min.apply(null, ribcageValues);
    // let maxRibVal = Math.max.apply(null, ribcageValues);
    let ribcageRange = maxRibVal - minRibVal;
    var ribcagePlotValues = ribcageValues.map(function(element) {
        return (element - minRibVal)/ribcageRange;
    });

    // if (ribcagePlotValues.length > 455) {
    if (ribcagePlotValues.length > 200) {
        ribcageValues.splice(0, 7);
    }
    drawWaves(ribcagePlotValues, ribcageCanvas, 1, 6.0);

    // Predicting airflow
    if (ribcageValues.length > 50 ){

        // let inputValues = ribcageValues.slice(-51, -1).map(function(element) {
            // return (

        fetch('http://127.0.0.1:5000/getEstimation', {
          method: 'post',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({'value': ribcageValues.slice(-52, -1)})
        })
        .then((response) => response.json()) .then((data) => {
          console.log('Success:', data);
            
            let airflow = Number(data.airflow);

            if (airflow < 0) {
                airflow = 0;
            }

            recentAirflow.push(airflow);

            let recentMax = Math.max.apply(null, recentAirflow);
            console.log(recentMax);

            airflowValues.push(recentMax);


            if (airflow > maxAirVal) {
                maxAirVal = airflow;
            }
            if (airflow < minAirVal) {
                minAirVal = airflow;
            }
            let airflowRange = maxAirVal - minAirVal;

            var airflowPlotValues = airflowValues.map(function(element) {
                return (element - minAirVal)/airflowRange;
            });

            // let a = airflowPlotValues.slice(-1);
            // a = a*200 - 100;

            airflowText.innerHTML = "Estimated airflow: " + airflowValues.slice(-1) + " l/min";
            // airflowText.innerHTML = "Predicted airflow: " + Math.round(a);

            // drawWaves(airflowPlotValues, airflowCanvas, 1, 42, 70);
            drawWaves(airflowPlotValues, airflowCanvas, 1, 60, 70);
                
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        
    } 

    // if (airflowValues.length > 64) {
    if (airflowValues.length > 20) {
        airflowValues.shift();
    }
    if (recentAirflow.length > 5) {
        recentAirflow.shift();
    }
}

function handleFlowAbdomenNotifications(event) {
    let value = event.target.value;
    let id = event.target.service.device.id;
    let int16View = new Int16Array(value.buffer);
    let timestamp = new Date().getTime();
    // TextDecoder to process raw data bytes.
    for (let i = 0; i < 7; i++) {

        let v = int16View[i];

        if (v > maxAbdVal) {
            maxRibVal = v;
        }
        if (v < minAbdVal) {
            minRibVal = v;
        }

        abdomenValues.push(int16View[i]);
    }
    abdomenText.innerHTML = "Abdomen movement: " + int16View[0].toString() + " mV";
    
    // let minRibVal = Math.min.apply(null, ribcageValues);
    // let maxRibVal = Math.max.apply(null, ribcageValues);
    let abdomenRange = maxAbdVal - minAbdVal;
    var abdomenPlotValues = abdomenValues.map(function(element) {
        return (element - minAbdVal)/abdomenRange;
    });

    // if (ribcagePlotValues.length > 455) {
    if (abdomenPlotValues.length > 200) {
        abdomenValues.splice(0, 7);
    }
    drawWaves(abdomenPlotValues, abdomenCanvas, 1, 6.0);

    // Predicting airflow
    if (abdomenValues.length > 51 ){

        // let inputValues = ribcageValues.slice(-51, -1).map(function(element) {
            // return (

        fetch('http://127.0.0.1:5000/getEstimation', {
          method: 'post',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({'value': ribcageValues.slice(-52, -1)})
        })
        .then((response) => response.json()) .then((data) => {
          console.log('Success:', data);
            
            let airflow = Number(data.airflow);

            if (airflow < 0) {
                airflow = 0;
            }

            recentAirflow.push(airflow);

            let recentMax = Math.max.apply(null, recentAirflow);
            console.log(recentMax);

            airflowValues.push(recentMax);


            if (airflow > maxAirVal) {
                maxAirVal = airflow;
            }
            if (airflow < minAirVal) {
                minAirVal = airflow;
            }
            let airflowRange = maxAirVal - minAirVal;

            var airflowPlotValues = airflowValues.map(function(element) {
                return (element - minAirVal)/airflowRange;
            });

            // let a = airflowPlotValues.slice(-1);
            // a = a*200 - 100;

            airflowText.innerHTML = "Estimated airflow: " + airflowValues.slice(-1) + " l/min";
            // airflowText.innerHTML = "Predicted airflow: " + Math.round(a);

            // drawWaves(airflowPlotValues, airflowCanvas, 1, 42, 70);
            drawWaves(airflowPlotValues, airflowCanvas, 1, 60, 70);
                
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        
    } 

    // if (airflowValues.length > 64) {
    if (airflowValues.length > 20) {
        airflowValues.shift();
    }
    if (recentAirflow.length > 5) {
        recentAirflow.shift();
    }
}
