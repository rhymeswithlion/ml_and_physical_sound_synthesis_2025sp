
import("stdfaust.lib");

average(x)	= (x + x') / 2.0;

process = vgroup("impulse", (button("play"): ba.impulsify))
		: vgroup("resonator", (+ ~ (de.delay(4096, 5) : average)));
