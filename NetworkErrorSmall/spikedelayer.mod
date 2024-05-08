COMMENT
SpikeDelayer

Relay signals after a normally distributed delay
ENDCOMMENT

NEURON	{ 
  ARTIFICIAL_CELL SpikeDelayer
  RANGE mean_delay, std_delay
}

PARAMETER {
	mean_delay	= 0 (ms)
	std_delay   = 0 (ms)
}

ASSIGNED {
	event (ms)
}

PROCEDURE seed(x) {
	set_seed(x)
}

INITIAL {
}	


NET_RECEIVE (w) {
	event = t + normrand(mean_delay, std_delay)
	if (event < t) {
		event = t
	}
	net_event(event)
}

