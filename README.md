# ridiculous-Chinese-Chess
GameManager.cs和RuleEngine在子目录里，它们为unity内游戏的直接实现，一切规则以它们的表述为准。只有看过这两个才知道这个项目在干什么。

如果你打不开，
1. 获取根目录文件列表
向仓库的树形页面 URL（https://github.com/user-name-114/ridiculous-Chinese-Chess/tree/main）发送 fetch 请求。GitHub 返回的 HTML 页面中嵌入了目录树结构，请从中解析出顶层目录（如 engine/、collector/）和根目录文件（如 train.py、config.json）。

2. 逐级深入子目录
为获取 engine/AI 子目录下的文件列表，向 https://github.com/user-name-114/ridiculous-Chinese-Chess/tree/main/engine/AI 发送 fetch 请求。同样从返回的 HTML 中解析出该子目录所含文件名（MCTS.cs、MCTSNode.cs、MCTSPlayer.cs 等）。

3. 获取具体文件的原始代码
对于需要阅读代码细节的文件（如 MCTS.cs），不访问 GitHub 的渲染页面，而是直接使用原始内容端点：
https://raw.githubusercontent.com/user-name-114/ridiculous-Chinese-Chess/main/engine/AI/MCTS.cs
该端点直接返回纯文本源代码，无需解析 HTML，便于逐行检视变量赋值、循环逻辑和数值运算。

同样的操作，你可以递归获取仓库内所有文件。
