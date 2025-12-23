import docMap from './docs.json';


/**
 * Formats a JSDoc object into a human-readable string.
 * @private
 * @param {object} doc - The JSDoc object for a single symbol.
 * @returns {string} A formatted string representing the documentation.
 */
function formatDoc(doc) {
    // Start with the name and description
    let output = `[ ${doc.longname} ]\n\n`;
    if(doc.kind === 'class'){
        output = `[ class ${doc.longname} ]\n\n`;
    }

    output += doc.description || 'No description available.';
    output += '\n\n';

    // Add parameters section
    if (doc.params && doc.params.length > 0) {
        output += 'Parameters:\n';
        for (const param of doc.params) {
            const type = param.type ? `{${param.type.names.join('|')}}` : '';
            const optional = param.optional ? '[optional]' : '';
            const defaultValue = param.defaultvalue !== undefined ? `(default: ${JSON.stringify(param.defaultvalue)})` : '';
            output += `  - ${param.name} ${type} ${optional} ${defaultValue}\n`;
            if (param.description) {
                output += `    ${param.description||''}\n`;
            }
        }
        output += '\n';
    }

    // Add returns section
    if (doc.returns && doc.returns.length > 0) {
        output += 'Returns:\n';
        for (const ret of doc.returns) {
            const type = ret.type ? `{${ret.type.names.join('|')}}` : '';
            output += `  ${type} - ${ret.description}\n`;
        }
        output += '\n';
    }

    // Add examples section
    if (doc.examples) {
        output += 'Examples:\n';
        for (const example of doc.examples) {
            output += '```javascript\n';
            output += example + '\n';
            output += '```\n';
        }
    }

    return output;
}


/**
 * Formats a JSDoc object into an HTML string.
 * @private
 * @param {object} doc - The JSDoc object for a single symbol.
 * @returns {string} A formatted HTML string representing the documentation.
 */
function formatDocHTML(doc) {
    let output = `<div class="ndarray-help-container">`;

    // Header
    const kind = doc.kind === 'class' ? `<span class="ndarray-help-kind">${doc.kind}</span> ` : '';
    output += `<h3 class="ndarray-help-longname">[ ${kind}${doc.longname} ]</h3>`;

    // Description
    output += `<p class="ndarray-help-description">${doc.description || 'No description available.'}</p>`;

    // Parameters
    if (doc.params && doc.params.length > 0) {
        output += `<div class="ndarray-help-parameters"><h4>Parameters:</h4><ul>`;
        for (const param of doc.params) {
            const type = param.type ? `<span class="ndarray-help-param-type">{${param.type.names.join('|')}}</span>` : '';
            const optional = param.optional ? `<span class="ndarray-help-param-attrs">[optional]</span>` : '';
            const defaultValue = param.defaultvalue !== undefined ? `<span class="ndarray-help-param-attrs">(default: ${JSON.stringify(param.defaultvalue)})</span>` : '';
            output += `<li>
                <strong class="ndarray-help-param-name">${param.name}</strong>
                ${type} ${optional} ${defaultValue}
                ${param.description ? `<p class="ndarray-help-param-desc">${param.description}</p>` : ''}
            </li>`;
        }
        output += `</ul></div>`;
    }

    // Returns
    if (doc.returns && doc.returns.length > 0) {
        output += `<div class="ndarray-help-returns"><h4>Returns:</h4>`;
        for (const ret of doc.returns) {
            const type = ret.type ? `<span class="ndarray-help-return-type">{${ret.type.names.join('|')}}</span>` : '';
            output += `<p>${type} - ${ret.description||''}</p>`;
        }
        output += `</div>`;
    }

    // Examples
    if (doc.examples) {
        output += `<div class="ndarray-help-examples"><h4>Examples:</h4>`;
        for (const example of doc.examples) {
            output += `<pre><code class="language-javascript">${example}</code></pre>`;
        }
        output += `</div>`;
    }

    output += `</div>`;
    return output;
}

/**
 * @namespace help
 */
export const help={
    /**
     * A WeakMap that maps live function/class objects back to their documentation names.
     * This map is populated by the registration logic in `index.js`.
     * @type {WeakMap<object, string>}
     */
    helpmap : new WeakMap(),


    /**
     * Returns help doc object
     * @param {string | object} target - The name of the function/class, or the object itself.
     * @returns {object | undefined} The documentation obj, or undefined if not found.
     */
    getHelpDoc(target) {
        let name;
        if (typeof target === 'string') {
            name = target;
        } else if ((typeof target === 'object' && target !== null) || typeof target === 'function') {
            name = help.helpmap.get(target);
        }
        if(!name){
            return ;
        }

        const doc = docMap[name];

        return doc;
    },


    /**
     * Provides help for a given class/function name or a live object.
     * Logs the formatted documentation to the console.
     * @param {string | object} target - The name of the function/class, or the object itself.
     */
    helpdoc(target) {
        const doc = help.getHelpDoc(target);

        if (!doc) {
            const targetName = typeof target === 'string' ? target : 'the provided object';
            console.log(`No documentation found for '${targetName}'.`);
            
            if (typeof target !== 'string') {
                console.log("Ensure the object is part of the ndarray library and the help system is initialized correctly.");
            }
            
            console.log('\nAvailable top-level names:\n' + Object.keys(docMap).filter(k => !k.includes('.')).sort().join('\n'));
            return;
        }
        return formatDoc(doc);
    },


    /**
     * Returns help content as a styled HTML string for a given class/function name or a live object.
     * @param {string | object} target - The name of the function/class, or the object itself.
     * @returns {string | undefined} The documentation as an HTML string, or undefined if not found.
     */
    helphtml(target) {
        const doc = help.getHelpDoc(target);

        if (!doc) {        
            return ;
        }
        return formatDocHTML(doc);
    }
};
