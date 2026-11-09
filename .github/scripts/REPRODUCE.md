# 手动复现 HCU CI

在 Linux HCU 节点上，使用此分支中的脚本部署某次 CI 的构建环境。CI 容器照常删除；只有主动执行本脚本才创建复现容器，且不会自动删除。

## 使用

节点需要 Python 3.10+、Git、Docker（当前用户有访问权限）、GitHub CLI `gh`，以及 HCU 驱动和 `/dev/kfd`、`/dev/dri`、`/opt/hyhal`。需要能访问 GitHub、记录的镜像仓库、内部 PyPI 和 AICC 下载地址。先按节点要求登录镜像仓库；需要访问 CI artifacts 时先执行 `gh auth login`。

下载含有复现脚本的分支后执行（`RUN_ID` 换成功能上线后的一次运行编号）：

```bash
bash .github/scripts/reproduce-ci.sh \
  --repo HYGON-AI/flash-train \
  --run-id RUN_ID \
  --torch-version 2.13.0 \
  --step build
```

该命令下载匹配矩阵的元数据，恢复源码并校验 tree SHA，按 digest 拉取镜像，创建容器，安装精确版本 AICC/Python 依赖，执行与 CI 相同的编译及 `CIUpload REPAIR`。**不会执行 NIGHTLY 上传或发布记录。**

- 不传 `--step`：部署后进入交互式 Bash，退出后停止容器但不删除。
- `--step setup`：只部署依赖，结束后停止并保留容器。
- `--step build`：部署、编译和修复 wheel 版本，结束后停止并保留容器。
- `--step test`：先编译并修复 wheel，再执行该矩阵的 C++/Python 测试；仅适用于原本启用了测试的任务。
- `--attempt 1`：指定该运行的重试次数；默认选择匹配 Torch 版本的最新可用元数据。
- `--work-dir /data/reproduce/run-123-torch2130`：指定一个尚不存在的本地目录。默认是当前目录下 `.ci-reproduce/<run>-<attempt>-torch<version>/`。不会覆盖已有目录。
- `--manifest /path/to/manifest.json`：使用手动下载的清单，仍需传入对应 `--repo`、`--run-id`、`--torch-version`；这种方式不需要 `gh`。将同一个 artifact 中的 `hcu_llvm.info` 放在清单旁边。

若 CI 在依赖安装阶段失败，清单会标记 `setup_complete: false`。此时只允许部署已记录的部分环境或交互排查，不把它声称为已经完整安装成功的环境；如果连 AICC 安装包地址/校验值都未采集到，则明确拒绝自动部署。初始化镜像/checkout 之前的失败不支持一键复现。

## 进入和删除

脚本结束或失败时会打印容器名、目录和管理命令；也会保存 `container-name.txt` 和 `enter.sh`：

```bash
# 重新启动容器、加载 common.sh 并进入交互式终端
bash /data/reproduce/run-123-torch2130/enter.sh

# 也可以手动进入
docker start <容器名>
docker exec -it <容器名> bash
# 容器内：cd 到记录的工作目录，source .github/scripts/common.sh

# 手动停止和删除容器
docker stop <容器名>
docker rm <容器名>
```

复现容器不会配置 `--rm`，部署失败、构建失败或交互终端退出均不删除。容器独立于 CI，不挂载宿主机 Docker socket。与当前 HCU 构建任务一致，它使用设备映射和 privileged 模式，应在相应的开发/CI 节点运行。

源码、临时文件和日志挂载在自己的 `workspace/`、`tmp/`、`logs/` 目录，删除容器不会删除这些文件；源码也不会被 runner 下一次 checkout 覆盖。环境装在容器可写层，保留该容器才能继续使用已经安装的依赖。重新运行完整构建会清理 `build-wheel`，如需排查中间文件应先检查再构建。

## CI 记录的信息及边界

`_build-test-wheel.yml` 在 checkout 后采集初始信息，在后续步骤失败后也尽量采集、上传 `reproduce-info-<run>-<attempt>-dtk2610-py3.12-torch2.13.0` artifact，保留 30 天。

- 来源/目标仓库 URL、实际合并提交的两个父 SHA、实际 tree SHA。恢复时重新合并并校验 tree；合并提交 SHA 可以不同。
- 正在运行的基础镜像的 registry digest；不使用可变 tag 代替，不保存容器快照。
- AICC 实际选定的 `.run` URL、完整版本和 SHA-256。安装前采集，再安装同一份文件；复现时重新下载并校验，不选择新的 nightly 包。
- 实际安装的 Python 包版本，包括 Torch 的完整 vendor tag。先从 DTK 对应源安装 Torch，再按记录版本安装其他依赖并校验。被测试的 `flash-train` wheel 不当成外部依赖下载，而由源码重新构建。
- 明确列出的构建变量、矩阵开关、失败阶段和 AICC 安装记录。不保存 GitHub Token、PyPI 密码或完整环境变量转储。

只支持产生了上述元数据的新任务，历史 run 无法自动补齐精确版本。远端提交、历史镜像或包必须仍可下载；缺失、checksum/tree 不一致时会报错停止，不回退到最新版。未保存源码归档，不保存 CI 原来的编译缓存或进程状态。

节点驱动、HCU 型号和外部网络/服务仍由复现节点提供，因此不保证复现临时网络故障或硬件差异。发布环境暂不纳入此功能。

## 本地开发检查

```bash
python3 -m unittest discover -s .github/scripts/tests -p 'test_reproduce_ci.py' -v
```

这些是离线测试：覆盖固定 SHA 重建/校验、环境清单和 artifact 选择，以及通过 mock Docker 检查容器的创建、失败后保留和手动进入。真实 HCU 节点验证仍需手动执行一次。
