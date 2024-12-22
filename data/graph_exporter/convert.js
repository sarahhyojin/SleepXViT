const path = require('path');
const glob = require('glob');
const edfExport = require('./edf');
const config = require('./config.json');
const fs = require('fs');

const type = process.env.TYPE;

const files = glob.sync(config.inputDir + '/*.edf', {nocase: true});

const main = async () => {
    for (file of files) {
        console.log(file);
        const filename = path.basename(file);
        const outputPath = path.join(config.outputDir, filename.replace('.edf', '.edf_done'))

        
        // 이미 파일이 존재하면 건너뛰기
        if (fs.existsSync(outputPath)) {
            console.log(`Output file already exists for ${filename}. Skipping...`);
            continue;
        }
  
        await edfExport.startExport(file, filename, type);
        
    }
};

main();
