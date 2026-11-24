import json
import paddle
import psutil

assert paddle.is_compiled_with_cuda(), 'Not a CUDA Paddle build'
assert paddle.device.cuda.device_count() > 0
paddle.set_device('gpu:0')
paddle.seed(42)
net = paddle.nn.Linear(8, 4)
opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=net.parameters())
before = net.weight.detach().clone()
loss = (net(paddle.randn([4, 8])) ** 2).mean()
loss.backward()
assert bool(paddle.isfinite(loss).item())
assert net.weight.grad is not None and bool(paddle.isfinite(net.weight.grad).all().item())
opt.step()
assert bool((net.weight != before).any().item())
paddle.utils.run_check()
print(json.dumps(dict(paddle=paddle.__version__, cuda=paddle.version.cuda(),
                      device=paddle.device.get_device(), loss=float(loss.item()),
                      backward_update='passed', cpu_count=psutil.cpu_count(),
                      ram_total=psutil.virtual_memory().total, ram_available=psutil.virtual_memory().available)))
