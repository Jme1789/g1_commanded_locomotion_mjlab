# Optional GetUp model placeholder

The compatible GetUp `policy.onnx` is intentionally not redistributed in this
repository because the external reference does not provide an explicit model
redistribution license.

Fallen damping protection works without this file. If recovery is requested
while the model is absent or invalid, GetUp initialization fails safely and the
FSM remains in or returns to Fallen.

Users who independently obtain a compatible model and have permission to use it
may place it here as:

~~~text
policy.onnx
~~~

The model must match
`../params/deploy.yaml`, including joint order and observation/action
dimensions. Always validate recovery with the robot suspended and an emergency
stop available.
