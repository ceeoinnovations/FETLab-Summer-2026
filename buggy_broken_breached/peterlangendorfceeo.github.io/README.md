# Buggy, Broken, or Breached!

*A 2-player local web game of programming, sabotage, and mechanical deduction.*

### Live Demos
* [**Standard Game**](https://peterlangendorfceeo.github.io/BuggyBrokenBreached/src)
* [**Speed Round**](https://peterlangendorfceeo.github.io/BuggyBrokenBreached/src/?mode=speed_round) (For demos)
---

## The Game Explained

**Buggy, Broken, or Breached!** is an adversarial puzzle game played on a single screen alongside a live LEGO robot. It is a game of deduction where players must figure out exactly why a robot didn't do what it was told.

In this game, there are three forces at play: The **Programmer**, the **Hacker**, and the **Hardware**. 

The Programmer wants the robot to execute a specific path. The Hacker secretly steps in to alter that path. But the biggest twist is the **Hardware itself**: the robot is highly unreliable. As it drives, there is a constant, looming threat of **Spontaneous Motor Failures**. At any moment, the left or right wheel might physically break down, dropping to 0% power for exactly one instruction before recovering.

When the robot finally finishes moving, it will likely be way off course. The programmer must then deduce the fate of *every single instruction* using telemetry graphs and cryptographic hashes. 

For every move in the sequence, you must answer one question:
* Was the move **VALID**? (The robot executed the original code perfectly).
* Was the move **BREACHED**? (The Hacker secretly replaced or deleted the code).
* Was the move **BROKEN**? (The code was safe, but the physical motor failed and stopped spinning).
* Or was it **BOTH**? (The hacker replaced the code **AND** the motor failed).

---

## The Phases of Play

### Phase 1: Programmer Input
The **Programmer** uses the terminal to build a sequence of movements (Forward, Back, Left, Right) and sets the exact speeds and angles for the motors. 
* *Note: You do NOT need to memorize your sequence! Your original instructions are securely backed up and will be visible to you during the final investigation phase.*

### Phase 2: Hacker Interception
The Programmer looks away, and the **Hacker** takes the keyboard. They have a limited number of edits (based on difficulty) to secretly **Replace** or **Delete** the Programmer's instructions. The goal is to be subtle enough to confuse the Programmer, making them wonder if the robot just broke down naturally.

### Phase 3: Hardware Link & Execution
The Bluetooth motors spin up and execute the final, hacked payload. This is where the **Motor Failures** happen in real-time. Watch the robot closely—if you see it drag a wheel or fail to turn, a motor just broke!

### Phase 4: The Incident Report
The robot stops, and the investigation begins. Players must use their detective tools to fill out the final Incident Report, marking each step as Valid, Breached, or Broken.

---

## Investigation Tools

To figure out what happened, you are given two highly restricted diagnostic tools:

1. **The Sensor Graph (Find the "BROKEN" moves):** This graph plots the actual speed of the Left and Right motors over time. You are looking for anomalies (specifically, flatlines). If an instruction tells the robot to move, but the graph shows a motor dropping to zero, you have successfully identified a mechanical motor failure.
   
2. **The Secure Memory Audit (Find the "BREACHED" moves):** This tool displays the Programmer's *original* instructions side-by-side with the robot's memory. You can select an instruction to audit its cryptographic hash. 
   * Flashes **GREEN**: The hashes match. The code is safe.
   * Flashes **RED**: The hashes differ. You just caught the Hacker!
   * *Beware: You a limited amount of guesses per game! Use them strategically on the moves you are most suspicious of.*

---

## Difficulty Levels

The game features dynamic difficulties that scale the complexity of the puzzle, limit the Hacker's power, and encrypt the diagnostic tools:

* **EASY:**
   * Maintinance workers treated like kings (motor failiure chance 5%)
   * Programmer has carpal tunnel (5-10 instructions)
   * Hacker connects to coffee shop wifi (max 5 changes)
   * Programmer is tech savvy (5 hashes, full graph)
   * Programmer arrived early to work (2m/2m/5m timers)
   * *1x multiplier*
* **MEDIUM:**
   * Maintinance workers paid minimum wage (motor failiure chance 10%)
   * Programmer types 120 words per minute (9-14 instructions)
   * Hacker connects with secure wifi (max 7 changes)
   * Programmer can handle technology (4 hashes, partial graph)
   * Programmer got to work just on time (1.5m/1.5m/4m timers)
   * *2x multiplier*
* **HARD:**
   * Maintinance workers fired (motor failiure chance 20%)
   * Programmer vibe codes (12-16 instructions)
   * Hacker uses satalite connections (unlimited changes)
   * Programmer is a grandma (3 hashes, limited graph)
   * Programmer is late to work (1m/1m/3m timers)
   * *3x multiplier*

---

## Hosting

This game runs entirely in the browser using HTML, CSS, JavaScript, and PyScript for the Bluetooth backend. It requires absolutely no installation or local servers for the players.

Simply click the link, pair your Web-Bluetooth compatible LEGO DoubleMotor, and start hacking! 

*(Note: Web Bluetooth requires a compatible modern browser, such as Google Chrome or Microsoft Edge).*
