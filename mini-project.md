
write abstract until 16.12.2016

battery health state based on discharge time

liquid state machine

investigate: http://stats.stackexchange.com/a/133292/141912
             explaining the numerical instability
             IRLS (max likelihood methods) can not deal with many perfect solutions -> instability

do not use max pooling on Atari! Makes local translation invariance -> ignores time / movement

NOT FORGET:
Theory: removing avg. reward samples from experience

We tried to simulate "n-step Q" method (shown to us by Bellec) by propagating 
the reward (after a point is scored in pong) several frames backwards. Also we 
have to learn for the length of the steps that lead to point times, to simulate 
learning after every action. See paper "n step Q async methods"

Similar suggestion from the playing atari Nature paper is using the 
"prioritized sweeping" which is basically sorting the memory tries by the 
error and discarding the least interesting ones for learning (the ones with 
lowest error) instead of oldest ones.

