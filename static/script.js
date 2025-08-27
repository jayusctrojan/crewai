// CrewAI Studio JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('studioForm');
    const runButton = document.getElementById('runButton');
    const clearButton = document.getElementById('clearButton');
    const outputContent = document.getElementById('outputContent');
    const statusIndicator = document.getElementById('statusIndicator');
    const executionInfo = document.getElementById('executionInfo');
    const loadingModal = document.getElementById('loadingModal');

    // Form elements
    const agentName = document.getElementById('agentName');
    const agentRole = document.getElementById('agentRole');
    const agentGoal = document.getElementById('agentGoal');
    const taskDescription = document.getElementById('taskDescription');
    const expectedOutput = document.getElementById('expectedOutput');
    const llmModel = document.getElementById('llmModel');

    // Execution info elements
    const infoAgentName = document.getElementById('infoAgentName');
    const infoModel = document.getElementById('infoModel');
    const infoStatus = document.getElementById('infoStatus');
    const infoDuration = document.getElementById('infoDuration');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await runAgent();
    });

    // Clear form handler
    clearButton.addEventListener('click', function() {
        form.reset();
        clearOutput();
    });

    // Auto-save form data to localStorage
    const formInputs = [agentName, agentRole, agentGoal, taskDescription, expectedOutput, llmModel];
    formInputs.forEach(input => {
        // Load saved data
        const savedValue = localStorage.getItem(input.id);
        if (savedValue) {
            input.value = savedValue;
        }

        // Save on change
        input.addEventListener('input', function() {
            localStorage.setItem(input.id, input.value);
        });
    });

    async function runAgent() {
        const startTime = Date.now();

        try {
            // Validate form
            if (!validateForm()) {
                return;
            }

            // Show loading state
            setLoadingState(true);
            updateStatus('running', 'Agent is working...');

            // Prepare request data
            const requestData = {
                agent_name: agentName.value.trim(),
                agent_role: agentRole.value.trim(),
                agent_goal: agentGoal.value.trim(),
                task_description: taskDescription.value.trim(),
                expected_output: expectedOutput.value.trim(),
                llm_model: llmModel.value
            };

            // Make API request
            const response = await fetch('/studio/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            const duration = Date.now() - startTime;

            if (response.ok && result.success) {
                // Success
                displayResult(result, duration);
                updateStatus('success', 'Completed successfully');
            } else {
                // Error from API
                throw new Error(result.detail || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('Error running agent:', error);
            displayError(error.message);
            updateStatus('error', 'Execution failed');
        } finally {
            setLoadingState(false);
        }
    }

    function validateForm() {
        const requiredFields = [
            { field: agentName, name: 'Agent Name' },
            { field: agentRole, name: 'Agent Role' },
            { field: agentGoal, name: 'Agent Goal' },
            { field: taskDescription, name: 'Task Description' },
            { field: expectedOutput, name: 'Expected Output' }
        ];

        for (let { field, name } of requiredFields) {
            if (!field.value.trim()) {
                alert(`Please fill in the ${name} field.`);
                field.focus();
                return false;
            }
        }
        return true;
    }

    function setLoadingState(loading) {
        runButton.disabled = loading;
        if (loading) {
            runButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Agent...';
            loadingModal.style.display = 'flex';
        } else {
            runButton.innerHTML = '<i class="fas fa-play"></i> Run Agent';
            loadingModal.style.display = 'none';
        }
    }

    function updateStatus(type, message) {
        statusIndicator.className = `status-indicator ${type}`;
        statusIndicator.innerHTML = `<i class="fas fa-circle"></i> ${message}`;
    }

    function displayResult(result, duration) {
        // Clear placeholder
        outputContent.innerHTML = '';

        // Create result container
        const resultContainer = document.createElement('div');
        resultContainer.className = 'result-content';
        resultContainer.textContent = result.result;

        outputContent.appendChild(resultContainer);

        // Update execution info
        infoAgentName.textContent = result.agent_name || agentName.value;
        infoModel.textContent = result.model_used || llmModel.value;
        infoStatus.textContent = 'Success';
        infoDuration.textContent = formatDuration(duration);

        executionInfo.style.display = 'block';

        // Scroll to output
        outputContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function displayError(errorMessage) {
        // Clear placeholder
        outputContent.innerHTML = '';

        // Create error container
        const errorContainer = document.createElement('div');
        errorContainer.style.cssText = `
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 4px solid var(--error-color);
            border-radius: 8px;
            padding: 1.5rem;
            color: var(--error-color);
        `;

        errorContainer.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error</strong>
            </div>
            <div>${errorMessage}</div>
        `;

        outputContent.appendChild(errorContainer);

        // Update execution info
        infoStatus.textContent = 'Error';
        infoAgentName.textContent = agentName.value || '-';
        infoModel.textContent = llmModel.value || '-';
        infoDuration.textContent = '-';

        executionInfo.style.display = 'block';
    }

    function clearOutput() {
        outputContent.innerHTML = `
            <div class="placeholder">
                <i class="fas fa-robot"></i>
                <p>Configure your agent above and click "Run Agent" to see the results here.</p>
            </div>
        `;
        executionInfo.style.display = 'none';
        updateStatus('', 'Ready');
        
        // Clear localStorage
        formInputs.forEach(input => {
            localStorage.removeItem(input.id);
        });
    }

    function formatDuration(ms) {
        if (ms < 1000) {
            return `${ms}ms`;
        } else if (ms < 60000) {
            return `${(ms / 1000).toFixed(1)}s`;
        } else {
            const minutes = Math.floor(ms / 60000);
            const seconds = ((ms % 60000) / 1000).toFixed(0);
            return `${minutes}m ${seconds}s`;
        }
    }

    // Add some example presets
    function loadPreset(preset) {
        switch(preset) {
            case 'marketing':
                agentName.value = 'Marketing Specialist';
                agentRole.value = 'Senior Marketing Manager';
                agentGoal.value = 'Create compelling marketing content that drives engagement and conversions';
                taskDescription.value = 'Write a product launch announcement for a new AI-powered project management tool';
                expectedOutput.value = 'A professional press release with key features, benefits, and call-to-action';
                break;
            case 'support':
                agentName.value = 'Customer Support Agent';
                agentRole.value = 'Customer Success Manager';
                agentGoal.value = 'Provide helpful and empathetic customer support solutions';
                taskDescription.value = 'Draft a response to a customer complaint about delayed delivery';
                expectedOutput.value = 'A professional, empathetic response with solution and compensation offer';
                break;
        }
        
        // Trigger save
        formInputs.forEach(input => {
            localStorage.setItem(input.id, input.value);
        });
    }

    // Expose preset function globally for potential future use
    window.loadPreset = loadPreset;

    console.log('CrewAI Studio initialized successfully!');
});