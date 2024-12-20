const fs = require('fs');
const path = require('path');
const EdfExporter = require('./EdfExporter');
const config = require('../config.json');

const edfExporter = new EdfExporter();

const outputDir = config.outputDir;

const jobController = (inputFile, inputId,  callback) => {

    let jobCount = 0;
    let isDone = false;

    const finish = () => isDone = true;

    const increaseJob = () => jobCount += 1;

    const decreaseJob = () => {

        jobCount -= 1;

        if(isDone && jobCount === 0) {

            try {

                console.log(`${inputId} | Done!`);

                // fs.unlinkSync(inputFile); // delete edf file

                // rename finished directory
                const currentDir = path.join(outputDir, `${inputId}`);
                const newDir = path.join(outputDir, `${inputId}_done`);
                fs.renameSync(currentDir, newDir);

                callback();

            } catch (error) {

                console.log(error);

            }

        }

    };

    return {
        finish: finish,
        increaseJob: increaseJob,
        decreaseJob: decreaseJob
    };

};

const startExport = (inputFile, inputId, inputType) => {
    return new Promise((resolve) => {
        let config;
        if(inputType === 'nx') config = require('../config/config_nx.json');
        else if(inputType === 'em') config = require('../config/config_em.json');
		else if(inputType === 'shhs1') config = require('../config/config_shhs1.json');
        else if (inputType == 'physionet2018') config = require('../config/config_physionet2018.json');
        else if(inputType === 'shhs2') config = require('../config/config_shhs2.json');
        else if(inputType === 'shhs1dup') config = require('../config/config_shhs1dup.json');
        const jobControllerInstance = jobController(inputFile, inputId, () => resolve());
    
        edfExporter.exportFile(inputFile, inputId, outputDir, config, jobControllerInstance);
    });
};

module.exports = {
    startExport: startExport
};
