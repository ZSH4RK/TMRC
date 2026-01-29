<img width="512" height="512" alt="TMRClogo" src="https://github.com/user-attachments/assets/6821a4b4-25c0-4267-ad84-d8776ba37e0e" />

# TMRC
The Trinity Machine Racing Championship is a competition run at Trinity School, to see who can create the fastest AI bot in Trackmania Nations Forever. The competitors will have access to the TMInterface game state (eg. Acceleration, Velocity, Floor contact per wheel, Wall contact flag), the .gbx map file and a compressed image, that can be used for path tracing. The model will be submitted as a python agent.py file to be run by a central server that runs all models in equal conditions for the overall race result.


## Plan 
1-3 Proof of Concept <br>
MVP 4-5 <br>
Nice To Have 6- <br>

- [ ] TMNF running with TMInterface loading onto a specific map (A01) from Commandline
- [ ] Get a basic model that just drives forward running from Command Line
- [ ] Use Docker to package the model and allow participants to use their own models
- [ ] Logging mechanism for results eg. Distance Covered or Racetime, Timestamp, inputs performed (In the form of a TMNF Replay file)
- [ ] Allow results to be sent back to host Computer and saved
- [ ] Live Leaderboard on each track
- [ ] Parallelize running TMNF while allowing the model to learn from all instances at once

### Technologies
* Python
* TMInterface
* Docker
* Commandline
