/**
 * Gamepad Control System for LaserCar
 * Handles gamepad input and communicates with server for controlling
 * motors, laser, and UI elements
 */

class GamepadController {
    constructor() {
        this.connected = false;
        this.gamepad = null;
        this.buttonState = {};
        this.axisState = {};
        this.activeMotor = 0; // 0 = Motor 1, 1 = Motor 2
        this.selectedSlider = 0;
        this.sliders = ['power', 'duration', 'safety_delay'];
        this.autoModeSliders = ['power', 'aim_power', 'duration', 'safety_delay'];
        this.currentSliders = this.sliders;
        this.autoMode = false;
        this.pollInterval = null;
        this.deadZone = 0.15;
        this.lastAxis0 = 0;
        this.lastAxis1 = 0;
        
        // Motor control parameters
        this.motorBaseSpeed = 100;
        this.motorMaxSpeed = 200;
        this.motorSpeed = this.motorBaseSpeed;
        this.motorAcceleration = 5;
        this.lastMotorMoveTime = 0;
        this.continuousMove = false;
        this.moveDirection = { x: 0, y: 0 };
        
        // Constants
        this.BUTTONS = {
            A: 0,           // Start experiment in auto mode
            B: 1,           // Exit/cancel
            X: 2,           // Toggle manual/auto mode
            Y: 3,           // Confirm high power
            LB: 4,          // Not used
            RB: 5,          // Switch active motor
            LT: 6,          // Toggle laser
            RT: 7,          // Not used
            BACK: 8,        // Not used
            START: 9,       // Not used
            L_STICK: 10,    // Not used
            R_STICK: 11,    // Not used
            DPAD_UP: 12,    // Move Y-
            DPAD_DOWN: 13,  // Move Y+
            DPAD_LEFT: 14,  // Move X-
            DPAD_RIGHT: 15, // Move X+
        };
        
        // Motor control boundaries
        this.MOTOR_MIN = 80;
        this.MOTOR_MAX = 4040;
        this.MOTOR_CENTER = 2048;
        
        // Initialize
        this.init();
    }
    
    init() {
        // Check for gamepad support
        if (!navigator.getGamepads) {
            console.error("Gamepad API not supported in this browser");
            return;
        }
        
        // Setup event listeners
        window.addEventListener("gamepadconnected", this.onGamepadConnected.bind(this));
        window.addEventListener("gamepaddisconnected", this.onGamepadDisconnected.bind(this));
        
        // Check if gamepad is already connected
        this.checkGamepads();
        
        // Start polling loop if no connection events
        if (!this.connected) {
            this.startPolling();
        }
        
        // Start continuous movement loop for smoother motor control
        setInterval(() => {
            if (this.continuousMove) {
                this.updateMotorMovement();
            }
        }, 50); // 20 times per second
    }
    
    checkGamepads() {
        const gamepads = navigator.getGamepads();
        for (let i = 0; i < gamepads.length; i++) {
            if (gamepads[i] && gamepads[i].connected) {
                this.gamepad = gamepads[i];
                this.connected = true;
                console.log("Found connected gamepad:", this.gamepad.id);
                this.startPolling();
                return true;
            }
        }
        return false;
    }
    
    onGamepadConnected(event) {
        console.log("Gamepad connected:", event.gamepad.id);
        this.gamepad = event.gamepad;
        this.connected = true;
        this.startPolling();
    }
    
    onGamepadDisconnected(event) {
        console.log("Gamepad disconnected:", event.gamepad.id);
        this.connected = false;
        this.gamepad = null;
        this.stopPolling();
    }
    
    startPolling() {
        if (this.pollInterval !== null) return;
        
        this.pollInterval = setInterval(() => {
            this.pollGamepad();
        }, 50); // 20 times per second
    }
    
    stopPolling() {
        if (this.pollInterval !== null) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        
        // Stop motor movement when controller disconnects
        this.continuousMove = false;
        this.stopMotor();
    }
    
    pollGamepad() {
        if (!this.connected) {
            // Try to find a gamepad
            if (!this.checkGamepads()) return;
        }
        
        // Get fresh gamepad data
        const gamepads = navigator.getGamepads();
        this.gamepad = gamepads[this.gamepad.index];
        
        if (!this.gamepad) {
            console.log("Gamepad disconnected during polling");
            this.connected = false;
            this.stopPolling();
            return;
        }
        
        // Process buttons
        this.processButtons();
        
        // Process axes
        this.processAxes();
    }
    
    processButtons() {
        // Initialize button state on first run
        if (Object.keys(this.buttonState).length === 0) {
            for (let i = 0; i < this.gamepad.buttons.length; i++) {
                this.buttonState[i] = false;
            }
        }
        
        // Check each button for state change
        for (let i = 0; i < this.gamepad.buttons.length; i++) {
            const pressed = this.gamepad.buttons[i].pressed;
            
            // Button state changed from not pressed to pressed
            if (pressed && !this.buttonState[i]) {
                this.handleButtonDown(i);
            } 
            // Button state changed from pressed to not pressed
            else if (!pressed && this.buttonState[i]) {
                this.handleButtonUp(i);
            }
            
            // Update state
            this.buttonState[i] = pressed;
        }
    }
    
    processAxes() {
        // Apply deadzone
        let axes = this.gamepad.axes.map(value => 
            Math.abs(value) < this.deadZone ? 0 : value
        );
        
        // Left stick (0, 1)
        const lx = axes[0];
        const ly = axes[1];
        
        // Filter out small movements to prevent jitter
        if (Math.abs(lx - this.lastAxis0) > 0.1 || Math.abs(ly - this.lastAxis1) > 0.1) {
            this.lastAxis0 = lx;
            this.lastAxis1 = ly;
            
            // Left stick vertical - change selected slider
            if (ly > 0.5) {
                // Down - next slider
                this.selectedSlider = (this.selectedSlider + 1) % this.currentSliders.length;
                this.highlightSlider();
            } else if (ly < -0.5) {
                // Up - previous slider
                this.selectedSlider = (this.selectedSlider - 1 + this.currentSliders.length) % this.currentSliders.length;
                this.highlightSlider();
            }
            
            // Left stick horizontal - adjust slider value
            if (lx !== 0) {
                this.adjustSelectedSlider(lx);
            }
        }
    }
    
    handleButtonDown(button) {
        console.log("Button pressed:", button);
        
        switch (button) {
            case this.BUTTONS.X:
                // Toggle between Manual and Auto mode
                this.toggleMode();
                break;
                
            case this.BUTTONS.Y:
                // Confirm high power
                this.confirmHighPower();
                break;
                
            case this.BUTTONS.A:
                // Start experiment in auto mode
                if (this.autoMode) {
                    this.runExperiment();
                }
                break;
                
            case this.BUTTONS.LT:
                // Toggle laser
                this.toggleLaser();
                break;
                
            case this.BUTTONS.RB:
                // Switch active motor
                this.switchMotor();
                break;
                
            case this.BUTTONS.DPAD_UP:
                // Start continuous movement up
                this.startContinuousMove(0, -1);
                break;
                
            case this.BUTTONS.DPAD_DOWN:
                // Start continuous movement down
                this.startContinuousMove(0, 1);
                break;
                
            case this.BUTTONS.DPAD_LEFT:
                // Start continuous movement left
                this.startContinuousMove(-1, 0);
                break;
                
            case this.BUTTONS.DPAD_RIGHT:
                // Start continuous movement right
                this.startContinuousMove(1, 0);
                break;
        }
    }
    
    handleButtonUp(button) {
        // Handle button release events if needed
        switch (button) {
            case this.BUTTONS.DPAD_UP:
            case this.BUTTONS.DPAD_DOWN:
            case this.BUTTONS.DPAD_LEFT:
            case this.BUTTONS.DPAD_RIGHT:
                // Stop motor movement when d-pad released
                this.stopContinuousMove();
                break;
        }
    }
    
    // UI and Control Functions
    
    highlightSlider() {
        const sliderName = this.currentSliders[this.selectedSlider];
        
        // Remove highlight from all sliders
        document.querySelectorAll('.slider-container').forEach(container => {
            container.style.backgroundColor = '';
        });
        
        // Highlight the selected slider
        const sliderId = `${sliderName === 'aim_power' ? 'aim-power' : sliderName}-slider`;
        const container = document.getElementById(sliderId).closest('.slider-container');
        if (container) {
            container.style.backgroundColor = 'rgba(100, 255, 218, 0.2)';
        }
    }
    
    adjustSelectedSlider(direction) {
        const sliderName = this.currentSliders[this.selectedSlider];
        let sliderId;
        
        switch (sliderName) {
            case 'power':
                sliderId = 'power-slider';
                break;
            case 'duration':
                sliderId = 'duration-slider';
                break;
            case 'safety_delay':
                sliderId = 'delay-slider';
                break;
            case 'aim_power':
                sliderId = 'aim-power-slider';
                break;
            default:
                return;
        }
        
        const slider = document.getElementById(sliderId);
        if (!slider) return;
        
        // Calculate step size based on slider range
        const range = slider.max - slider.min;
        const stepSize = Math.max(1, Math.round(range / 50));
        
        // Calculate new value
        let newValue = parseInt(slider.value) + (direction > 0 ? stepSize : -stepSize);
        newValue = Math.max(parseInt(slider.min), Math.min(parseInt(slider.max), newValue));
        
        // Update slider
        slider.value = newValue;
        
        // Trigger change event
        const event = new Event('input', { bubbles: true });
        slider.dispatchEvent(event);
        
        // Send command to server
        this.sendSliderCommand(sliderName, newValue);
    }
    
    toggleMode() {
        // Click the appropriate mode button
        if (this.autoMode) {
            document.getElementById('manual-mode-btn').click();
            this.autoMode = false;
            this.currentSliders = this.sliders;
        } else {
            document.getElementById('auto-mode-btn').click();
            this.autoMode = true;
            this.currentSliders = this.autoModeSliders;
        }
        
        // Reset selected slider
        this.selectedSlider = 0;
        this.highlightSlider();
    }
    
    confirmHighPower() {
        // Click the high power button
        const highPowerBtn = document.getElementById('high-power-btn');
        if (highPowerBtn && highPowerBtn.style.display !== 'none') {
            highPowerBtn.click();
        }
    }
    
    runExperiment() {
        // Click the experiment button
        document.getElementById('experiment-btn').click();
    }
    
    toggleLaser() {
        // Toggle laser
        document.getElementById('laser-toggle-btn').click();
    }
    
    switchMotor() {
        this.activeMotor = this.activeMotor === 0 ? 1 : 0;
        console.log(`Switched to Motor ${this.activeMotor + 1}`);
        
        // Visual feedback
        const motorLabel = document.createElement('div');
        motorLabel.style.position = 'fixed';
        motorLabel.style.top = '50%';
        motorLabel.style.left = '50%';
        motorLabel.style.transform = 'translate(-50%, -50%)';
        motorLabel.style.background = 'rgba(0,0,0,0.7)';
        motorLabel.style.color = this.activeMotor === 0 ? '#64ffda' : '#FF9500';
        motorLabel.style.padding = '20px';
        motorLabel.style.borderRadius = '10px';
        motorLabel.style.fontSize = '24px';
        motorLabel.style.fontWeight = 'bold';
        motorLabel.style.zIndex = '9999';
        motorLabel.innerText = `Motor ${this.activeMotor + 1} Active`;
        
        document.body.appendChild(motorLabel);
        
        setTimeout(() => {
            document.body.removeChild(motorLabel);
        }, 1500);
        
        // Send command to server for motor switching
        this.sendCommand('motor', { motor: this.activeMotor, action: 'select' });
    }
    
    startContinuousMove(x, y) {
        // Set movement direction
        this.moveDirection = { x, y };
        this.continuousMove = true;
        this.motorSpeed = this.motorBaseSpeed;
        this.lastMotorMoveTime = Date.now();
        
        // Initial movement
        this.updateMotorMovement();
    }
    
    stopContinuousMove() {
        this.continuousMove = false;
        this.moveDirection = { x: 0, y: 0 };
        this.motorSpeed = this.motorBaseSpeed;
        this.stopMotor();
    }
    
    updateMotorMovement() {
        if (!this.continuousMove) return;
        
        // Calculate time since last move for acceleration
        const now = Date.now();
        const deltaTime = (now - this.lastMotorMoveTime) / 1000; // In seconds
        this.lastMotorMoveTime = now;
        
        // Gradually increase speed for longer movement
        this.motorSpeed = Math.min(this.motorSpeed + (this.motorAcceleration * deltaTime * 60), this.motorMaxSpeed);
        
        // Calculate movement step based on direction and current speed
        const xStep = this.moveDirection.x * this.motorSpeed;
        const yStep = this.moveDirection.y * this.motorSpeed;
        
        // Adjust the movement logic based on active motor
        if (this.activeMotor === 0) {
            // Motor 1: Standard mapping
            this.moveMotor(xStep, yStep);
        } else {
            // Motor 2: XY swapped orientation as per original code
            this.moveMotor(yStep, xStep);
        }
    }
    
    moveMotor(x, y) {
        // Send motor command to server
        this.sendCommand('motor', { 
            motor: this.activeMotor, 
            cmd: 'move',
            dx: x,
            dy: y
        });
    }
    
    stopMotor() {
        // Send stop command to server
        this.sendCommand('motor', { 
            motor: this.activeMotor, 
            cmd: 'get' // Just get the current position, effectively stopping
        });
    }
    
    // Server Communication
    
    sendCommand(type, value) {
        fetch('/gamepad-control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: type,
                value: value
            })
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                console.error('Command failed:', data.error);
            }
        })
        .catch(err => {
            console.error('Error sending command:', err);
        });
    }
    
    sendSliderCommand(slider, value) {
        let cmd = '';
        
        switch (slider) {
            case 'power':
                cmd = 'set_power';
                break;
            case 'duration':
                cmd = 'set_duration';
                break;
            case 'safety_delay':
                cmd = 'set_safety_delay';
                break;
            case 'aim_power':
                cmd = 'set_aim_power';
                break;
        }
        
        if (cmd) {
            fetch('/laser-cmd', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cmd, value })
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    console.error('Slider command failed:', data.error);
                }
            })
            .catch(err => {
                console.error('Error sending slider command:', err);
            });
        }
    }
}

// Initialize controller when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.gamepadController = new GamepadController();
    
    // Check for gamepad connections every 3 seconds
    setInterval(() => {
        if (!window.gamepadController.connected) {
            window.gamepadController.checkGamepads();
        }
    }, 3000);
}); 