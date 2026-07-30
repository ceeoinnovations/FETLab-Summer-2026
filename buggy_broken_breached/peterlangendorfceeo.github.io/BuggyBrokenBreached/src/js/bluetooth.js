window.legoBluetooth = {
    device: null,
    client: null,
    writeChar: null,
    notifyChar: null,
    _pending: {},   

    // --- THE GATT MUTEX ---
    _writeQueue: [],
    _isWriting: false,

    // Confirmed via chrome://bluetooth-internals against the real hardware.
    SERVICE_UUID: '0000fd02-0000-1000-8000-00805f9b34fb',
    WRITE_UUID:   '0000fd02-0001-1000-8000-00805f9b34fb',
    NOTIFY_UUID:  '0000fd02-0002-1000-8000-00805f9b34fb',

    INFO_REQUEST: 0,
    INFO_RESPONSE: 1,
    MOTOR_RUN_COMMAND: 122,
    MOTOR_RUN_RESULT: 123,
    MOTOR_RUN_FOR_DEGREES_COMMAND: 124,
    MOTOR_RUN_FOR_DEGREES_RESULT: 125,
    MOTOR_STOP_COMMAND: 138,
    MOTOR_STOP_RESULT: 139,
    MOTOR_SET_SPEED_COMMAND: 140,
    MOTOR_SET_SPEED_RESULT: 141,
    DEVICE_NOTIFICATION_REQUEST: 40,
    DEVICE_NOTIFICATION_RESPONSE: 41,
    DEVICE_NOTIFICATION: 60,

    SUB_NOTIFICATION_SIZES: { 0: 2, 1: 20, 3: 3, 4: 1, 10: 12, 12: 12, 15: 6, 16: 1 },
    IMU_DEVICE_NOTIFICATION: 1,

    MOTOR_BITS_LEFT: 1,
    MOTOR_BITS_RIGHT: 2,
    MOTOR_BITS_BOTH: 3,

    DIRECTION_CLOCKWISE: 0,
    DIRECTION_COUNTERCLOCKWISE: 1,
    DIRECTION_SHORTEST: 2,
    DIRECTION_LONGEST: 3,

    STATUS_LABEL: {0: "COMPLETED", 1: "INTERRUPTED", 2: "NACK"},

    _rawYawTenthDeg: null,
    zeroAngle: null,

    imu: {
        orientation: null, yawFace: null,
        yaw: null, pitch: null, roll: null,
        accelX: null, accelY: null, accelZ: null,
        gyroX: null, gyroY: null, gyroZ: null,
    },

    motorTelemetry: {},


    connectHub: async function() {
        try {
            console.log("Requesting Web Bluetooth Connection...");

            // MY FIX: Re-enabled the strict Double Motor filter
            this.device = await navigator.bluetooth.requestDevice({
                filters: [{ services: [this.SERVICE_UUID] }]
            });

            this.client = await this.device.gatt.connect();
            await new Promise(resolve => setTimeout(resolve, 500));

            const service = await this.client.getPrimaryService(this.SERVICE_UUID);
            this.writeChar = await service.getCharacteristic(this.WRITE_UUID);
            this.notifyChar = await service.getCharacteristic(this.NOTIFY_UUID);

            await this.notifyChar.startNotifications();
            this.notifyChar.addEventListener('characteristicvaluechanged', (event) => {
                this._handleNotification(new Uint8Array(event.target.value.buffer));
            });

            console.log("Sending InfoRequest handshake...");
            const infoResponse = await this._sendAndWait(
                new Uint8Array([this.INFO_REQUEST]), this.INFO_RESPONSE
            );
            const rpcMajor = infoResponse[0], rpcMinor = infoResponse[1];
            const rpcBuild = infoResponse[2] | (infoResponse[3] << 8);
            console.log(`Handshake complete. Device RPC version: ${rpcMajor}.${rpcMinor}.${rpcBuild}`);

            await this.enableTelemetry(50);

            // MY FIX: Return device name for UI footer
            return this.device.name || "DOUBLE MOTOR";

        } catch (error) {
            console.error("Web Bluetooth Error: ", error);
            return false;
        }
    },

    enableTelemetry: async function(delayMs = 50) {
        const req = new Uint8Array([this.DEVICE_NOTIFICATION_REQUEST, delayMs & 0xFF, (delayMs >> 8) & 0xFF]);
        await this._sendAndWait(req, this.DEVICE_NOTIFICATION_RESPONSE);
        const start = Date.now();
        while (this._rawYawTenthDeg === null && Date.now() - start < 2000) {
            await new Promise(r => setTimeout(r, 20));
        }
        if (this._rawYawTenthDeg !== null) {
            this.zeroAngle = this._rawYawTenthDeg / 10.0;
            console.log(`IMU telemetry live. Zeroed starting angle at ${this.zeroAngle.toFixed(1)}deg.`);
        } else {
            console.warn("No IMU telemetry received -- angle-feedback turning will not be available.");
        }
    },

    normalizeAngle: function(angle) {
        if (angle > 180) return angle - 360;
        if (angle < -180) return angle + 360;
        return angle;
    },

    getAngle: function() {
        if (this._rawYawTenthDeg === null || this.zeroAngle === null) return null;
        let angle = (this._rawYawTenthDeg / 10.0) - this.zeroAngle;
        if (angle > 0) angle = angle - 360;
        angle = (angle + 180) * -1;
        if (angle < 0) angle = angle + 180;
        else if (angle > 0) angle = angle - 180;
        return Math.round(angle * 10) / 10;
    },

    getGyro: function() {
        return {
            raw: { x: this.imu.gyroX, y: this.imu.gyroY, z: this.imu.gyroZ },
            degPerSec: {
                x: this.imu.gyroX === null ? null : this.imu.gyroX / 10.0,
                y: this.imu.gyroY === null ? null : this.imu.gyroY / 10.0,
                z: this.imu.gyroZ === null ? null : this.imu.gyroZ / 10.0,
            }
        };
    },

    getImuSnapshot: function() {
        return { ...this.imu };
    },

    getMotorSpeed: function(bitMask) {
        const t = this.motorTelemetry[bitMask];
        return t ? t.speed : null;
    },

    getMotorTelemetry: function(bitMask) {
        return this.motorTelemetry[bitMask] || null;
    },

    _handleNotification: function(bytes) {
        const msgType = bytes[0];
        const payload = bytes.slice(1);

        if (msgType === this.DEVICE_NOTIFICATION) {
            this._parseDeviceNotification(payload);
            return;
        }

        const queue = this._pending[msgType];
        if (queue && queue.length > 0) {
            const resolve = queue.shift();
            resolve(payload);
        }
    },

    _parseDeviceNotification: function(payload) {
        if (payload.length < 2) return;
        let dataLen = payload[0] | (payload[1] << 8);
        let data = payload.slice(2, 2 + dataLen);

        while (data.length > 0) {
            const subType = data[0];
            const size = this.SUB_NOTIFICATION_SIZES[subType];
            if (size === undefined) break; 

            if (subType === this.IMU_DEVICE_NOTIFICATION) {
                const view = new DataView(data.buffer, data.byteOffset + 1, size);
                this._rawYawTenthDeg = view.getInt16(2, true);
                this.imu.orientation = view.getUint8(0);
                this.imu.yawFace = view.getUint8(1);
                this.imu.yaw = view.getInt16(2, true);
                this.imu.pitch = view.getInt16(4, true);
                this.imu.roll = view.getInt16(6, true);
                this.imu.accelX = view.getInt16(8, true);
                this.imu.accelY = view.getInt16(10, true);
                this.imu.accelZ = view.getInt16(12, true);
                this.imu.gyroX = view.getInt16(14, true);
                this.imu.gyroY = view.getInt16(16, true);
                this.imu.gyroZ = view.getInt16(18, true);
            } else if (subType === 10 /* MOTOR_NOTIFICATION */) {
                const view = new DataView(data.buffer, data.byteOffset + 1, size);
                const bitMask = view.getUint8(0);
                this.motorTelemetry[bitMask] = {
                    motorState: view.getUint8(1),
                    absolutePosition: view.getUint16(2, true),
                    power: view.getInt16(4, true),
                    speed: view.getInt8(6),
                    position: view.getInt32(7, true),
                    gesture: view.getInt8(11),
                    updatedAt: Date.now(),
                };
            }

            data = data.slice(1 + size);
        }
    },

    // MY FIX: Adds the Mutex loop to pace transmissions and prevent radio crashes
    _processWriteQueue: async function() {
        if (this._isWriting || this._writeQueue.length === 0) return;
        this._isWriting = true;
        const task = this._writeQueue.shift();
        
        try {
            await this.writeChar.writeValueWithoutResponse(task.bytes);
            await new Promise(r => setTimeout(r, 40)); 
            task.resolve();
        } catch (e) {
            task.reject(e);
        }
        
        this._isWriting = false;
        this._processWriteQueue();
    },

    _safeWrite: function(bytes) {
        return new Promise((resolve, reject) => {
            this._writeQueue.push({ bytes, resolve, reject });
            this._processWriteQueue();
        });
    },

    _rawWrite: async function(bytes) {
        // MY FIX: Forces Claude's raw payload to go through the Mutex Queue
        await this._safeWrite(bytes);
    },

    // MY FIX: Added blocking argument to bypass await timeouts
    _sendAndWait: function(bytes, expectedResultType, timeoutMs = 4000, blocking = true) {
        if (!blocking) {
            return this._rawWrite(bytes).then(() => null);
        }
        const waiter = this._waitForResult(expectedResultType, timeoutMs);
        return this._rawWrite(bytes).then(() => waiter).catch((err) => { throw err; });
    },

    // MY FIX: Prevents unhandled promises by resolving to null instead of throwing errors
    _waitForResult: function(expectedResultType, timeoutMs = 4000) {
        return new Promise((resolve, reject) => {
            if (!this._pending[expectedResultType]) this._pending[expectedResultType] = [];
            const timer = setTimeout(
                () => resolve(null), 
                timeoutMs
            );
            this._pending[expectedResultType].push((payload) => {
                clearTimeout(timer);
                resolve(payload);
            });
        });
    },

    _buildSetSpeed: function(bitMask, speed) {
        return new Uint8Array([this.MOTOR_SET_SPEED_COMMAND, bitMask, speed & 0xFF]);
    },

    _buildRunForDegrees: function(bitMask, degrees, direction) {
        return new Uint8Array([
            this.MOTOR_RUN_FOR_DEGREES_COMMAND,
            bitMask,
            degrees & 0xFF, (degrees >> 8) & 0xFF, (degrees >> 16) & 0xFF, (degrees >> 24) & 0xFF,
            direction
        ]);
    },

    _buildRun: function(bitMask, direction) {
        return new Uint8Array([this.MOTOR_RUN_COMMAND, bitMask, direction]);
    },

    _buildStop: function(bitMask) {
        return new Uint8Array([this.MOTOR_STOP_COMMAND, bitMask]);
    },

    _concat: function(...arrays) {
        const total = arrays.reduce((sum, a) => sum + a.length, 0);
        const out = new Uint8Array(total);
        let offset = 0;
        for (const a of arrays) { out.set(a, offset); offset += a.length; }
        return out;
    },

    runMotorForDegrees: async function(bitMask, speedPercent, direction, degrees) {
        const combined = this._concat(
            this._buildSetSpeed(bitMask, speedPercent),
            this._buildRunForDegrees(bitMask, degrees, direction)
        );
        return this._sendAndWait(combined, this.MOTOR_RUN_FOR_DEGREES_RESULT);
    },

    runForDegreesSynced: async function(commands) {
        const parts = [];
        for (const c of commands) {
            parts.push(this._buildSetSpeed(c.bitMask, c.speedPercent));
            parts.push(this._buildRunForDegrees(c.bitMask, c.degrees, c.direction));
        }
        const combined = this._concat(...parts);
        const waiters = commands.map(() => this._waitForResult(this.MOTOR_RUN_FOR_DEGREES_RESULT));
        await this._rawWrite(combined);
        return Promise.all(waiters);
    },

    runMotorContinuous: async function(bitMask, speedPercent, direction) {
        const combined = this._concat(
            this._buildSetSpeed(bitMask, speedPercent),
            this._buildRun(bitMask, direction)
        );
        // MY FIX: Set blocking=false to prevent Timeouts during while loop
        return this._sendAndWait(combined, this.MOTOR_RUN_RESULT, 4000, false);
    },

    stopMotor: async function(bitMask) {
        // MY FIX: Set blocking=false
        return this._sendAndWait(this._buildStop(bitMask), this.MOTOR_STOP_RESULT, 4000, false);
    },

    turnUntilAngle: async function(bitMask, speedPercent, direction, targetAngle, toleranceDeg = 7.5, maxMs = 15000) {
        if (this.getAngle() === null) {
            console.warn("No IMU angle available; falling back is the caller's responsibility.");
            return false;
        }
        await this.runMotorContinuous(bitMask, speedPercent, direction);

        const start = Date.now();
        while (Date.now() - start < maxMs) {
            const current = this.getAngle();
            const error = Math.abs(this.normalizeAngle(current - targetAngle));
            if (error < toleranceDeg) break;
            await new Promise(r => setTimeout(r, 20));
        }
        await this.stopMotor(bitMask);
        return true;
    }
};