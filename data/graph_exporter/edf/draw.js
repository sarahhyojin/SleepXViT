const fs = require('fs');

//import * as d3 from 'd3';
const d3 = require('d3');

// 나머지 코드에서 d3를 사용

const { JSDOM } = require('jsdom');
const xmlSerializer = require('xmlserializer');
const sharp = require('sharp');


// JSDOM for D3
const { document } = new JSDOM('').window;
global.document = document;

const convertUnit = (standardUnit, edfUnit) => {

    let mult = 1.0;

    if (edfUnit === 'm' + standardUnit) mult = 1000.0; // milli
    else if (edfUnit === 'u' + standardUnit) mult = 1000000.0; // micro

    return mult;
};

const exportPlot = async (data, config, filePath, jobControllerInstance) => {

    jobControllerInstance.increaseJob();

    const svgString = plotSvg(data, config);

    if (config.filetype === 'svg') await exportSVG(svgString, filePath);else if (config.filetype === 'png') await exportPNG(svgString, config.size, filePath);

    jobControllerInstance.decreaseJob();
};

const plotSvg = (data, config) => {

    const { width, height } = config.size;
    const svg = d3.create('svg').attr('viewBox', [0, 0, width, height]);

    /* Fill the background */
    svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', 'black');

    const { groups } = data;
    const sampleWidth = data.duration / data.maxDuration * width;

    function normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    let yOffset = 0;

    for (let i = 0; i < groups.length; i += 1) {
        const samples = groups[i];
        let groupHeight = 0;
        const groupYOffset = yOffset;

        for (let j = 0; j < samples.length; j += 1) {
            const sample = samples[j];
            let { min: yMin, max: yMax, height: sampleHeight } = data.signalConfig[i][j];
            const yOffsetConst = yOffset; // To prevent no-loop-func lint error

            const mult = convertUnit(data.signalConfig[i][j].unit, data.signalConfig[i][j].edfUnit);
            yMin *= mult;
            yMax *= mult;

            const sampleData = [];
            if (sample.length / data.maxDuration < 100) {
                for (let k = 0; k < sample.length; k += 1) {
                    const s = sample[k];
                    sampleData.push({
                        t: k / (sample.length - 1) * sampleWidth,
                        value: yOffsetConst + (1 - normalize(s, yMin, yMax)) * sampleHeight
                    });
                    sampleData.push({
                        t: (k + 1) / (sample.length - 1) * sampleWidth,
                        value: yOffsetConst + (1 - normalize(s, yMin, yMax)) * sampleHeight
                    });
                }
            } else {
                for (let k = 0; k < sample.length; k += 1) {
                    const s = sample[k];
                    sampleData.push({
                        t: k / (sample.length - 1) * sampleWidth,
                        value: yOffsetConst + (1 - normalize(s, yMin, yMax)) * sampleHeight
                    });
                }
            }

            svg.append('path').datum(sampleData).attr('clip-path', `url(#line-clip-${i})`).attr('fill', 'none').attr('stroke', 'white').attr('stroke-width', config.strokeWidth).attr('shape-rendering', 'geometricPrecision').attr('d', d3.line().x(d => Math.round(d.t)).y(d => d.value));

            yOffset += sampleHeight;
            groupHeight += sampleHeight;
        }

        svg.append('clipPath').attr('id', `line-clip-${i}`).append('rect').attr('x', 0).attr('y', groupYOffset).attr('width', sampleWidth).attr('height', groupHeight);
    }

    return xmlSerializer.serializeToString(svg.node());
};

const exportSVG = (svgString, filePath) => {

    fs.writeFileSync(filePath, svgString);
};

const exportPNG = (svgString, size, filePath) => {
    return new Promise(async (resolve, reject) => {

        const svgPath = filePath + '.svg';

        fs.writeFileSync(svgPath, svgString);

        await sharp(svgPath).png().toFile(filePath);

        fs.unlinkSync(svgPath);

        resolve();
    });
};

module.exports = {
    exportPlot: exportPlot
};


//module.exports = draw;
