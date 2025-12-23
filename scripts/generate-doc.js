const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

try {
    console.log('Generating documentation map from JSDoc comments...');

    // Define paths
    const rootDir = path.resolve(__dirname, '..');
    const srcDir = path.join(rootDir, 'src');
    const outputFile = path.join(srcDir, 'docs.json');
    const jsdocBin = path.join(rootDir, 'node_modules', '.bin', 'jsdoc');

    // 1. Run jsdoc to get JSON output
    const command = `"${jsdocBin}" -X -r "${srcDir}"`;
    const jsonOutput = execSync(command, { encoding: 'utf8' , maxBuffer: 10 * 1024 * 1024 });

    // 2. Process the JSON into a lean map, keeping only necessary fields.
    const rawDocs = JSON.parse(jsonOutput);
    const docMap = {};

    for (const doc of rawDocs) {
        if (doc.longname && doc.comment && doc.kind !== 'package') {
            // Normalize names like NDArray#get to NDArray.prototype.get for consistency
            const name = doc.longname.replace(/#/g, '.prototype.');
            
            // Pluck only the fields we need for the help function
            docMap[name] = {
                longname: doc.longname,
                kind: doc.kind,
                description: doc.description,
                params: doc.params?.map(p => ({
                    name: p.name,
                    description: p.description,
                    type: p.type,
                    optional: p.optional,
                    defaultvalue: p.defaultvalue,
                })),
                returns: doc.returns,
                examples: doc.examples,
            };
        }
    }
    const jsonString = JSON.stringify(docMap, null, 2);

    // 3. Write the map to the destination file
    fs.writeFileSync(outputFile, jsonString);
    console.log(`Successfully generated documentation map at ${outputFile}`);

} catch (error) {
    console.error('Failed to generate documentation map:', error);
    process.exit(1);
}
