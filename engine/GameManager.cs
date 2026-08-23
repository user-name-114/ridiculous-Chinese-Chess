using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;  // 文件顶部添加

//unity游戏内容的直接实现，不直接用于训练AI。

public class GameManager : MonoBehaviour
{
    public enum GameMode { Local, AI, Online }
    public GameMode gameMode = GameMode.Local;
    private AIController aiController;
    private bool aiProcessing = false;
    private bool aiActing = false;
    private const int aiTeam = -1; // AI 执黑
    public GameObject mainMenuPanel; // Inspector 拖入 MainMenuPanel
    public GameObject piecePrefab;          // 拖入棋子预制体
    public GameObject cellPrefab;           // 拖入棋盘格子预制体
    public GameObject blueDotPrefab;        // 拖入蓝点预制体
    public GameObject explosionParticlePrefab;   // 拖入粒子预制体
    private List<GameObject> activeDots = new List<GameObject>();
    // 箭头预制体（在 Inspector 中拖入）
    public GameObject arrowUpPrefab;
    public GameObject arrowDownPrefab;
    public GameObject arrowLeftPrefab;
    public GameObject arrowRightPrefab;
    private float arrowDistance = 0.65f;
    private List<GameObject> activeArrows = new List<GameObject>();
    private Gamestate state;
    public LotteryEffectDiscription lotteryEffectDiscription;
    private Dictionary<Piece, GameObject> pieceMap = new Dictionary<Piece, GameObject>();
    private bool isGenerationMode = false;
    private PieceType currentGenerationType = PieceType.Pawn; // 默认测试用兵
    private int generationRemaining = 0;   // 剩余可生成次数
    private bool isDefectMode = false;
    private HashSet<Piece> defectTargets = new HashSet<Piece>();
    public Piece selectedPiece;                    // 当前选中的棋子
    private List<LegalMove> currentLegalMoves;      // 选中棋子的合法移动列表
    public bool isMoving = false;   // 是否正在执行移动动画，阻止重复点击
    private float moveSpeed = 12f;      // 棋子移动速度，单位：格/秒
    public GameObject infoPanelPrefab;          // 在 Inspector 中拖入信息面板预制体
    private GameObject infoPanelInstance;
    private TMP_Text infoText;
    private List<Gamestate> stateHistory = new List<Gamestate>();   // 最近6个状态快照
    private const int MaxHistory = 6;
    // 复活列表相关
    public GameObject reviveListPanel;          // 拖入 ReviveListPanel (ScrollView)
    public GameObject reviveListItemPrefab;     // 拖入列表项预制体
    public Button undoButton;   // 在 Inspector 中将悔棋按钮拖入
    public ButtonControl buttonControl;
    public List<Sprite> pieceSprites;           // 棋子图片列表，Inspector拖入

    private List<GameObject> reviveListItems = new List<GameObject>();
    private bool isReviveMode = false;          // 是否处于复活模式
                                                // 复活模式下的选中状态
    private Piece selectedRevivePiece = null;
    private List<Vector2Int> currentRevivePositions = new List<Vector2Int>();
    private bool isShowingGraveyard = false;    // 是否正在展示浏览列表
    private int showingGraveyardTeam = 0;       // 当前浏览的阵营（1红/-1黑）
                                                // 升级模式
    private bool isUpgradeMode = false;
    private List<Piece> upgradeTargets = new List<Piece>();
    private int currentUpgradeEffectLevel = 0;
    private int upgradesRemaining = 0;
    private bool isFreezeMode = false;
    private bool isDefrostMode = false;
    private bool isPostDefrostMove = false;  // 解冻后强制移动子状态
    private Piece defrostedPiece = null;     // 刚解冻的棋子引用
    private List<Piece> freezeTargets = new List<Piece>();
    private List<Piece> defrostTargets = new List<Piece>();
    public TMP_Text lotteryResultText;          // 拖入抽奖结果文本
    // 抽奖按钮（拖入 Inspector）
    public Button lotteryButton;
    public TMP_Text turnIndicatorText;
    private bool pendingHideDescription = false;
    void SaveCurrentState()
    {
        stateHistory.Insert(0, state.DeepClone());
        if (stateHistory.Count > MaxHistory)
            stateHistory.RemoveAt(stateHistory.Count - 1); // 移除最旧的
    }
    void GenerateAllPieces()
    {
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece piece = state[x, y];
                Vector3 worldPos = new Vector3(x, y, 0);
                //先摆格子
                GameObject cell = Instantiate(cellPrefab, worldPos, Quaternion.identity, transform);
                Cell cellScript = cell.GetComponent<Cell>();
                cellScript.x = x;
                cellScript.y = y;
                cellScript.manager = this;       // 确保 Cell 能回调 GameManager


                if (piece.type == PieceType.Empty) continue;
                // 生成棋子，坐标转换（红方在下 y 小，黑方在上 y 大）
                GameObject go = Instantiate(piecePrefab, worldPos, Quaternion.identity, transform);

                // 设置图片
                PieceView view = go.GetComponent<PieceView>();
                view.Setup(piece);
                // 墙需要恢复倒计角标（领域重建路径，无动画）
                if (piece is Wall wall)
                    view.SetupWallCountdown(wall.wallDuration);
                // 记录棋子与物体的映射（后续移动用）
                pieceMap[piece] = go;
            }
        }
    }

    void Start()
    {
        Application.targetFrameRate = 60;
        state = new Gamestate();
        state.prepareModeOn = false; // 调试：关闭准备模式
        GenerateAllPieces();
        SaveCurrentState();
        lotteryResultText.gameObject.SetActive(false);
        if (infoPanelPrefab != null)
        {
            Canvas canvas = FindObjectOfType<Canvas>();
            Transform parent = canvas != null ? canvas.transform : transform;
            infoPanelInstance = Instantiate(infoPanelPrefab, parent);
            infoPanelInstance.SetActive(false);
            infoText = infoPanelInstance.GetComponentInChildren<TMP_Text>();
        }
        // 初始隐藏列表和按钮
        if (reviveListPanel != null) reviveListPanel.SetActive(false);
        buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode); // 设置按钮初始状态

        // 显示主菜单或按保留模式直接开始
        if (pendingGameMode.HasValue)
        {
            StartGame((int)pendingGameMode.Value);
            pendingGameMode = null;
        }
        else
        {
            if (mainMenuPanel != null)
                mainMenuPanel.SetActive(true);
        }
    }

    private static GameMode? pendingGameMode;

    /// <summary>重启场景时保留当前游戏模式</summary>
    public static void RestartWithMode(GameMode mode)
    {
        pendingGameMode = mode;
        UnityEngine.SceneManagement.SceneManager.LoadScene(
            UnityEngine.SceneManagement.SceneManager.GetActiveScene().buildIndex);
    }

    public Gamestate GetState() => state;

    public void StartGame(int mode)
    {
        gameMode = (GameMode)mode;
        if (mainMenuPanel != null)
            mainMenuPanel.SetActive(false);

        if (gameMode == GameMode.AI)
        {
            aiController = new AIController();
            aiController.Initialize(aiTeam);
        }
    }

    //--------以下为辅助方法------------
    Vector3 GetWorldPos(int x, int y)
    {
        return new Vector3(x, y, 0);
    }
    /// <summary>选中一个棋子，计算并显示其所有合法移动蓝点</summary>
    public void SelectPiece(Piece piece)
    {
        selectedPiece = piece;

        // 清除旧蓝点和箭头
        ClearDots();
        ClearArrows();

        // 计算并绘制合法移动蓝点
        currentLegalMoves = new List<LegalMove>();
        for (int tx = state.leftBound; tx <= state.rightBound; tx++)
        {
            for (int ty = state.lowerBound; ty <= state.upperBound; ty++)
            {
                if (piece.IsLegalMove(tx, ty, state))
                {
                    currentLegalMoves.Add(new LegalMove(piece.thisx, piece.thisy, tx, ty));
                    Vector3 dotPos = GetWorldPos(tx, ty);
                    GameObject dot = Instantiate(blueDotPrefab, dotPos, Quaternion.identity, transform);
                    activeDots.Add(dot);
                }
            }
        }
        // ----- 狙击手箭头生成 -----
        if (piece is Pawn pawn && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3))  // 升级为狙击手或自爆狙击兵
        {
            Vector3 piecePos = GetWorldPos(piece.thisx, piece.thisy);

            // 上
            if (pawn.CanSnipeInDirection(0, 1, state))
                CreateSniperArrow(arrowUpPrefab, piecePos + new Vector3(0, arrowDistance, 0), 0, 1, pawn);
            // 下
            if (pawn.CanSnipeInDirection(0, -1, state))
                CreateSniperArrow(arrowDownPrefab, piecePos + new Vector3(0, -arrowDistance, 0), 0, -1, pawn);
            // 左
            if (pawn.CanSnipeInDirection(-1, 0, state))
                CreateSniperArrow(arrowLeftPrefab, piecePos + new Vector3(-arrowDistance, 0, 0), -1, 0, pawn);
            // 右
            if (pawn.CanSnipeInDirection(1, 0, state))
                CreateSniperArrow(arrowRightPrefab, piecePos + new Vector3(arrowDistance, 0, 0), 1, 0, pawn);
        }

        // 选中放大
        if (pieceMap.TryGetValue(piece, out GameObject go))
            go.transform.localScale = Vector3.one * 1.1f;
    }
    /// <summary>取消选中，清除蓝点和箭头</summary>
    public void DeselectPiece()
    {    // 恢复棋子大小
        if (selectedPiece != null && pieceMap.TryGetValue(selectedPiece, out GameObject go))
            go.transform.localScale = Vector3.one;

        selectedPiece = null;
        currentLegalMoves = null;
        ClearDots();
        ClearArrows();
    }
    // 辅助方法：清除所有蓝点
    void ClearDots()
    {
        foreach (GameObject dot in activeDots)
            Destroy(dot);
        activeDots.Clear();
    }
    private void CreateSniperArrow(GameObject prefab, Vector3 position, int dx, int dy, Pawn sniper)
    {
        if (prefab == null) return;
        GameObject arrow = Instantiate(prefab, position, Quaternion.identity, transform);

        SniperArrow script = arrow.GetComponent<SniperArrow>();
        if (script != null)
        {
            script.dx = dx;
            script.dy = dy;
            script.sniper = sniper;
            script.manager = this;
        }

        // 冷却时变暗
        SpriteRenderer sr = arrow.GetComponent<SpriteRenderer>();
        if (sr != null)
        {
            sr.color = sniper.sniperAvailable ? Color.white : new Color(0.6f, 0.6f, 0.6f, 0.9f);
        }

        activeArrows.Add(arrow);
    }
    /// <summary> 清除所有箭头 </summary>
    public void ClearArrows()
    {
        foreach (GameObject arrow in activeArrows)
            Destroy(arrow);
        activeArrows.Clear();
    }
    /// <summary>进入生成棋子模式，扫描棋盘显示所有合法位置蓝点</summary>
    void EndTurnAndUpdate()
    {
        //0. 延迟隐藏抽奖描述面板（等下次点击时再消失）
        if (lotteryEffectDiscription != null)
            pendingHideDescription = true;
        // 1. 切换行动方
        state.currentTeam = -state.currentTeam;
        UpdateTurnIndicator();

        if (state.prepareModeOn)
        {
            // 准备阶段：狙击手不冷却、冻结不自然恢复
            state.prepareLotteryCount++;
            if (state.prepareLotteryCount >= 10)
                state.prepareModeOn = false;
        }
        else
        {
            // 2. 更新新行动方的冻结状态（数据）
            state.UpdateFrozenTurns();

            // 3. 视觉同步：冻结棋子每回合颜色渐变恢复，自然解冻的棋子完全恢复
            for (int x = state.leftBound; x <= state.rightBound; x++)
            {
                for (int y = state.lowerBound; y <= state.upperBound; y++)
                {
                    Piece p = state[x, y];
                    if (!pieceMap.TryGetValue(p, out GameObject go)) continue;
                    PieceView view = go.GetComponent<PieceView>();
                    if (view == null) continue;

                    if (p.frozenTurns > 0)
                    {
                        if (!view.IsFrozen())
                            view.SetFrozen(true, p.upgradeLevel > 0);
                        else
                        {
                            view.FadeFreezeVisual();
                            view.TickCountdown();
                        }
                        // 记录到数据层，用于领域展开/收缩后恢复正确的冻结视觉
                        p.freezeTickCount++;
                    }
                    else if (view.IsFrozen())
                        view.SetFrozen(false, p.upgradeLevel > 0);
                }
            }

            // 4. 狙击冷却更新
            state.UpdateAllSniperCooldowns();

            // 5. 墙持续回合更新
            for (int x = state.leftBound; x <= state.rightBound; x++)
            {
                for (int y = state.lowerBound; y <= state.upperBound; y++)
                {
                    Piece p = state[x, y];
                    if (p is Wall wall)
                    {
                        wall.wallDuration--;
                        if (wall.wallDuration <= 0)
                        {
                            state[x, y] = Empty.Instance;
                            if (pieceMap.TryGetValue(p, out GameObject wallGO))
                            {
                                Destroy(wallGO);
                                pieceMap.Remove(p);
                            }
                        }
                        else if (pieceMap.TryGetValue(p, out GameObject wallGO))
                        {
                            wallGO.GetComponent<PieceView>().TickWallCountdown();
                        }
                    }
                }
            }
        }

        // 6. 保存悔棋历史
        SaveCurrentState();
    }
    void UpdateTurnIndicator()
    {
        if (turnIndicatorText != null)
            turnIndicatorText.text = state.currentTeam == 1 ? "红方行动" : "黑方行动";
    }
    //------------模式切换----------------
    //-----------------------------------
    public void EnterGenerationMode(PieceType type)
    {
        if (isGenerationMode) return;
        isGenerationMode = true;
        currentGenerationType = type;
        ClearDots();

        int team = state.currentTeam;
        bool anyValid = false;
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                if (PieceGenerator.IsPositionValidForGeneration(state, x, y, currentGenerationType, team))
                {
                    Vector3 pos = GetWorldPos(x, y);
                    GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
                    activeDots.Add(dot);
                    anyValid = true;
                }
            }
        }

        // 如果没有合法位置，直接退出并结束回合
        if (!anyValid)
        {
            ExitGenerationMode(true);
            return;
        }
    }
    /// <summary>退出生成模式，清除蓝点</summary>
    public void ExitGenerationMode(bool endTurn = false)
    {
        isGenerationMode = false;
        generationRemaining = 0;
        ClearDots();
        if (endTurn)
            EndTurnAndUpdate();
    }
    /// <summary>进入叛变模式：在敌方非将非墙棋子上绘制蓝点</summary>
    public void EnterDefectMode()
    {
        if (isDefectMode) return;
        isDefectMode = true;
        defectTargets.Clear();
        ClearDots();

        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty || p.type == PieceType.Wall) continue;
                if (p.thisTeam == state.currentTeam) continue; // 只选敌方
                if (p.type == PieceType.King) continue;         // 排除将
                defectTargets.Add(p);
                Vector3 pos = GetWorldPos(x, y);
                GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
                activeDots.Add(dot);
            }
        }
    }
    /// <summary>退出叛变模式</summary>
    public void ExitDefectMode()
    {
        isDefectMode = false;
        defectTargets.Clear();
        ClearDots();
    }
    /// <summary>进入复活模式（根据 currentTeam 自动展开列表，可点击交互）</summary>
    public void EnterReviveMode()
    {
        if (isReviveMode) return;
        isReviveMode = true;

        if (undoButton != null) undoButton.gameObject.SetActive(false);
        // 隐藏浏览列表（如果打开）
        HideGraveyardList();
        // 隐藏两个按钮
        buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);

        // 生成可点击的己方坟墓列表
        int team = state.currentTeam;
        List<Piece> graveyard = (team == 1) ? state.redGraveyard : state.blackGraveyard;
        // 坟墓为空：直接退出，切换回合
        if (graveyard.Count == 0)
        {
            isReviveMode = false;
            if (undoButton != null) undoButton.gameObject.SetActive(true);
            buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);
            EndTurnAndUpdate();
            return;
        }
        // 检查是否有任意棋子存在合法复活位置
        bool hasValidPosition = false;
        foreach (Piece piece in graveyard)
        {
            var positions = state.GetInitialPositions(piece.type, piece.thisTeam);
            foreach (var (x, y) in positions)
            {
                if (IsPositionValidForRevive(x, y, piece.thisTeam))
                {
                    hasValidPosition = true;
                    break;
                }
            }
            if (hasValidPosition) break;
        }
        // 无合法复活位置：直接退出，切换回合
        if (!hasValidPosition)
        {
            isReviveMode = false;
            if (undoButton != null) undoButton.gameObject.SetActive(true);
            buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);
            EndTurnAndUpdate();
            return;
        }
        Transform content = reviveListPanel.transform.Find("Viewport/Content");
        foreach (Piece piece in graveyard)
        {
            GameObject item = Instantiate(reviveListItemPrefab, content);
            ReviveListItem listItem = item.GetComponent<ReviveListItem>();
            listItem.Setup(piece, this, true); // 可点击
            reviveListItems.Add(item);
        }

        reviveListPanel.SetActive(true);
    }
    /// 判断指定位置是否可以作为复活目标格。
    /// 空格：合法；敌方棋子：合法（会击杀）；己方棋子：需友伤开启；墙：非法。
    private bool IsPositionValidForRevive(int x, int y, int team)
    {
        if (!state.IsValidPosition(x, y)) return false;

        Piece target = state[x, y];
        if (target.type == PieceType.Wall) return false;        // 墙不能覆盖
        if (target.type == PieceType.Empty) return true;        // 空格合法
        if (target.thisTeam != team) return true;               // 敌方合法
        return Piece.friendlyFire == 1;                         // 己方需友伤
    }
    /// <summary>复活模式中点击列表项，进入位置选择阶段</summary>
    public void OnRevivePieceSelected(Piece piece)
    {
        if (!isReviveMode) return;

        // 清除旧蓝点和选中状态
        ClearDots();
        selectedRevivePiece = piece;
        currentRevivePositions.Clear();

        // 获取初始位置并过滤合法格
        var positions = state.GetInitialPositions(piece.type, piece.thisTeam);
        foreach (var (x, y) in positions)
        {
            if (IsPositionValidForRevive(x, y, piece.thisTeam))
            {
                currentRevivePositions.Add(new Vector2Int(x, y));
                Vector3 dotPos = GetWorldPos(x, y);
                GameObject dot = Instantiate(blueDotPrefab, dotPos, Quaternion.identity, transform);
                activeDots.Add(dot);
            }
        }

        // 列表保持展开，不隐藏
    }
    /// <summary>退出复活模式</summary>
    public void ExitReviveMode()
    {
        isReviveMode = false;
        if (undoButton != null) undoButton.gameObject.SetActive(true);
        selectedRevivePiece = null;
        currentRevivePositions.Clear();
        ClearDots();
        ClearReviveListItems();
        reviveListPanel.SetActive(false);
        buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);
    }
    /// 获取指定类型棋子的最大升级等级
    int GetMaxUpgradeLevel(PieceType type)
    {
        switch (type)
        {
            case PieceType.Rook: return 1;
            case PieceType.Knight: return 1;
            case PieceType.Cannon: return 3;
            case PieceType.Bishop: return 3;
            case PieceType.Guard: return 1;
            case PieceType.King: return 2;
            case PieceType.Pawn: return 3;
            default: return 0;
        }
    }
    /// 进入升级模式：筛选可升级的己方棋子，若无可直接切换回合
    public void EnterUpgradeMode(PieceType type, int effectLevel, int count = 1)
    {
        if (isUpgradeMode) return;
        isUpgradeMode = true;
        currentUpgradeEffectLevel = effectLevel;
        upgradesRemaining = count;

        int team = state.currentTeam;
        int maxLevel = GetMaxUpgradeLevel(type);
        upgradeTargets.Clear();

        // 遍历棋盘，收集符合条件的己方棋子
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece piece = state[x, y];
                if (piece.type != type || piece.thisTeam != team) continue;
                // 筛选：未造成过当前升级效果，且叠加后不超过最大等级
                if (piece.upgradeLevel != effectLevel && piece.upgradeLevel + effectLevel <= maxLevel)
                {
                    upgradeTargets.Add(piece);
                }
            }
        }

        // 无可升级棋子：直接退出升级模式，切换回合
        if (upgradeTargets.Count == 0)
        {
            isUpgradeMode = false;
            EndTurnAndUpdate();
            return;
        }

        // 在符合条件的棋子上绘制蓝点
        ClearDots();
        foreach (Piece p in upgradeTargets)
        {
            Vector3 pos = GetWorldPos(p.thisx, p.thisy);
            GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
            activeDots.Add(dot);
        }
    }
    /// 退出升级模式
    void ExitUpgradeMode()
    {
        isUpgradeMode = false;
        upgradeTargets.Clear();
        ClearDots();
    }
    /// 御驾亲征：自动升级己方将，无需手动选择
    void AutoUpgradeKing()
    {
        int team = state.currentTeam;
        int maxLevel = GetMaxUpgradeLevel(PieceType.King);
        Piece king = null;
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.King && p.thisTeam == team)
                {
                    king = p;
                    break;
                }
            }
            if (king != null) break;
        }
        // 将不存在或已达最大等级：跳过，切换回合
        if (king == null || king.upgradeLevel >= maxLevel)
        {
            EndTurnAndUpdate();
            return;
        }
        king.Upgrade(king.upgradeLevel + 1);
        if (pieceMap.TryGetValue(king, out GameObject go))
            go.GetComponent<PieceView>().SetUpgraded(king.upgradeLevel);
        EndTurnAndUpdate();
    }
    void EnterFreezeMode()
    {
        int team = state.currentTeam;
        int enemyTeam = -team;
        freezeTargets.Clear();
        ClearDots();

        // 收集敌方所有非墙棋子（允许重复冻结）
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type != PieceType.Empty && p.type != PieceType.Wall && p.thisTeam == enemyTeam)
                    freezeTargets.Add(p);
            }
        }

        // 无合法目标：跳过，切换回合
        if (freezeTargets.Count == 0)
        {
            EndTurnAndUpdate();
            return;
        }

        // 进入冻结模式，绘制蓝点
        isFreezeMode = true;
        foreach (Piece p in freezeTargets)
        {
            Vector3 pos = GetWorldPos(p.thisx, p.thisy);
            GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
            activeDots.Add(dot);
        }
    }
    void EnterDefrostMode()
    {
        int team = state.currentTeam;
        defrostTargets.Clear();
        ClearDots();

        // 收集己方所有被冻结的棋子
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.thisTeam == team && p.frozenTurns > 0)
                    defrostTargets.Add(p);
            }
        }

        // 没有可解冻的棋子，直接切换回合
        if (defrostTargets.Count == 0)
        {
            EndTurnAndUpdate();
            return;
        }

        // 进入解冻模式，绘制蓝点
        isDefrostMode = true;
        foreach (Piece p in defrostTargets)
        {
            Vector3 pos = GetWorldPos(p.thisx, p.thisy);
            GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
            activeDots.Add(dot);
        }
    }
    //-----------------------------------
    //-----------抽奖效果-----------------
    //-----------------------------------
    public void OnLottery()
    {
        if (isMoving) return;
        if (!aiActing && gameMode == GameMode.AI && state.currentTeam == aiTeam) return;
        if (selectedPiece != null) DeselectPiece();
        if (isUpgradeMode || isReviveMode || isDefectMode || isGenerationMode ||
            isFreezeMode || isDefrostMode || isPostDefrostMove || isDefectMode) return;
        StartCoroutine(LotteryAnimation());
    }
    private string[] prizeNames = new string[]
{
    // 索引0不使用，1~40对应抽奖编号
    "", "赛車", "炮车", "迫击炮", "狙击手", "御驾亲征", "巨象", "小飞象", "自爆兵",
    "捅了老窝", "秦王绕柱", "连环马", "武士", "炮兵", "万马奔腾", "停車场",
    "叛变", "起死回生", "起死回生", "起死回生", "起死回生", "起死回生",
    "反转", "洪水", "冲锋号", "激光炮", "领域展开", "领域收缩",
    "冻结", "冻结", "冻结", "解冻", "解冻", "解冻", "解冻", "解冻",
    "未中奖", "未中奖", "未中奖", "未中奖", "未中奖"
};

    public void ApplyReverse()
    {
        // 数据层反转：交换所有棋子阵营并对称坐标
        LotteryEffects.Reverse(state);

        // 收集动画所需数据
        var items = new List<(Piece piece, GameObject go, Vector3 from, Vector3 to)>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Empty) continue;
                if (pieceMap.TryGetValue(p, out GameObject go))
                {
                    Vector3 from = go.transform.position;
                    Vector3 to = GetWorldPos(p.thisx, p.thisy);
                    items.Add((p, go, from, to));
                }
            }
        }

        StartCoroutine(ReverseAnimation(items));
    }
    public void ApplyFlood()
    {
        if (isMoving) return;
        isMoving = true;

        // 1. 记录河边棋子（Flood会清除棋盘引用）
        List<Piece> riversidePieces = new List<Piece>();
        for (int y = 4; y <= 5; y++)
        {
            for (int x = state.leftBound; x <= state.rightBound; x++)
            {
                Piece p = state[x, y];
                if (p != null && p.type != PieceType.Empty)
                    riversidePieces.Add(p);
            }
        }

        // 2. 播放洪水动画（不等待，与数据操作并行）
        StartCoroutine(FloodAnimation());

        // 3. 数据层：击杀河边棋子
        LotteryEffects.Flood(state);

        // 4. 视觉隐藏被击杀的棋子
        foreach (Piece p in riversidePieces)
        {
            if (p.isDead && pieceMap.TryGetValue(p, out GameObject go))
            {
                go.GetComponent<SpriteRenderer>().enabled = false;
                pieceMap.Remove(p);
            }
        }

        // 5. 处理自爆连锁：同一批自爆兵同时爆炸（并行）
        List<Coroutine> explosions = new List<Coroutine>();
        foreach (Piece p in riversidePieces)
        {
            if (p.isDead && p is Pawn pawn && pawn.canExplode)
                explosions.Add(StartCoroutine(ProcessExplosions(p)));
        }

        // 6. 回合切换、冷却更新、状态保存
        EndTurnAndUpdate();
        isMoving = false;
    }
    public void ApplyChargeBugle()
    {
        // 1. 收集所有兵（调用前快照）
        var pawnSnapshots = new List<(Pawn pawn, Vector3 oldWorldPos, GameObject go)>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                Piece p = state[x, y];
                if (p.type == PieceType.Pawn && pieceMap.TryGetValue(p, out GameObject go))
                {
                    pawnSnapshots.Add(((Pawn)p, go.transform.position, go));
                }
            }
        }

        // 记录调用前坟墓计数，便于找出新增死亡棋子
        int prevRedDead = state.redGraveyard.Count;
        int prevBlackDead = state.blackGraveyard.Count;

        // 2. 执行数据层
        LotteryEffects.ChargeBugle(state);

        // 3. 收集移动列表和死亡列表
        var moves = new List<(GameObject go, Vector3 from, Vector3 to)>();
        var deadGOs = new List<GameObject>();

        foreach (var (pawn, oldPos, go) in pawnSnapshots)
        {
            if (pawn.isDead)
            {
                // 该兵已死，加入待隐藏列表
                deadGOs.Add(go);
            }
            else
            {
                Vector3 newPos = GetWorldPos(pawn.thisx, pawn.thisy);
                if (Vector3.Distance(oldPos, newPos) > 0.01f)
                {
                    moves.Add((go, oldPos, newPos));
                }
            }
        }

        // 收集非兵死亡棋子（通过坟墓新增）
        List<Piece> newDeadPieces = new List<Piece>();
        // 红方新增
        for (int i = prevRedDead; i < state.redGraveyard.Count; i++)
        {
            Piece dead = state.redGraveyard[i];
            newDeadPieces.Add(dead);
        }
        // 黑方新增
        for (int i = prevBlackDead; i < state.blackGraveyard.Count; i++)
        {
            Piece dead = state.blackGraveyard[i];
            newDeadPieces.Add(dead);
        }
        foreach (Piece dead in newDeadPieces)
        {
            if (pieceMap.TryGetValue(dead, out GameObject go))
            {
                deadGOs.Add(go);
            }
        }

        StartCoroutine(ChargeBugleAnimation(moves, deadGOs));
    }
    public void ApplyLaserCannon()
    {
        int targetY = LotteryEffects.LaserCannon(state);
        StartCoroutine(LaserAnimation(targetY));
    }
    void ApplyBoardExpand()
    {
        if (state.isBoardExpanded) { EndTurnAndUpdate(); return; }
        isMoving = true;
        LotteryEffects.ExpandBoard(state);
        foreach (GameObject go in pieceMap.Values) Destroy(go);
        pieceMap.Clear();
        foreach (Transform child in transform)
        {
            if (child.GetComponent<Cell>() != null)
                Destroy(child.gameObject);
        }
        GenerateAllPieces();
        StartCoroutine(BoardExpandCameraAnimation());
    }
    void ApplyBoardShrink()
    {
        if (!state.isBoardExpanded) { EndTurnAndUpdate(); return; }
        isMoving = true;
        List<Piece> killed = LotteryEffects.ShrinkBoard(state);
        foreach (Piece p in killed)
        {
            if (pieceMap.TryGetValue(p, out GameObject go))
            {
                go.GetComponent<SpriteRenderer>().enabled = false;
                pieceMap.Remove(p);
            }
        }
        foreach (Piece p in killed)
        {
            if (p is Pawn pawn && pawn.canExplode)
                StartCoroutine(ProcessExplosions(p));
        }
        foreach (GameObject go in pieceMap.Values) Destroy(go);
        pieceMap.Clear();
        foreach (Transform child in transform)
        {
            if (child.GetComponent<Cell>() != null)
                Destroy(child.gameObject);
        }
        GenerateAllPieces();
        StartCoroutine(BoardShrinkCameraAnimation());
    }
    public void ApplyWall()
    {
        StartCoroutine(SpawnWallAnimation());
    }
    public void ApplyGeneration(PieceType type, int count)
    {
        generationRemaining = count;
        EnterGenerationMode(type);
    }
    public void ApplyTongLeLaoWo()
    {
        int team = state.currentTeam;
        int riverY = (team == 1) ? 4 : 5;

        // 记录河边调用前的棋子快照（用于找出新生成的兵）
        var beforePieces = new HashSet<Piece>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            Piece p = state[x, riverY];
            if (p.type != PieceType.Empty)
                beforePieces.Add(p);
        }

        // 数据层生成兵（可能击杀原有棋子，已在GeneratePawnsOnRiver里处理）
        PieceGenerator.GeneratePawnsOnRiver(state, team);

        // 收集新生成的兵（包括因覆盖而可能残留的旧棋子已被清理，board[x,y]现在是新兵）
        var newPawns = new List<(int x, GameObject go)>();
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            Piece p = state[x, riverY];
            if (p.type == PieceType.Pawn && !beforePieces.Contains(p))
            {
                // 为该兵创建GameObject并播放淡入动画
                Vector3 worldPos = GetWorldPos(x, riverY);
                GameObject go = Instantiate(piecePrefab, worldPos, Quaternion.identity, transform);
                PieceView view = go.GetComponent<PieceView>();
                view.Setup(p);
                SpriteRenderer sr = go.GetComponent<SpriteRenderer>();
                if (sr != null) { sr.color = new Color(1, 1, 1, 0); }
                pieceMap[p] = go;
                newPawns.Add((x, go));
            }
        }

        // 如果有被击杀的棋子，视觉上已经由GeneratePawnsOnRiver内的KillAndClear处理（目前KillAndClear没有视觉隐藏，需补）
        // 为安全，可在此处对beforePieces中且已死的棋子进行视觉隐藏
        foreach (Piece p in beforePieces)
        {
            if (p.isDead && pieceMap.TryGetValue(p, out GameObject deadGO))
            {
                deadGO.GetComponent<SpriteRenderer>().enabled = false;
                pieceMap.Remove(p);
            }
        }

        // 处理可能因击杀引发的自爆
        foreach (Piece p in beforePieces)
        {
            if (p.isDead && p is Pawn pawn && pawn.canExplode)
                StartCoroutine(ProcessExplosions(p));
        }

        StartCoroutine(TongLeLaoWoAnimation(newPawns));
    }
    //-------------动画-------------------
    /// 棋子到目标位置，更新棋盘状态，并切换回合。
    IEnumerator MoveToTarget(int x2, int y2)
    {
        isMoving = true;
        // 移动后自动收起正在浏览的失子列表
        if (isShowingGraveyard)
            HideGraveyardList();

        int oldLianHuanMaTeam = state.lianHuanMaTeam;

        LegalMove move = currentLegalMoves.Find(m => m.x2 == x2 && m.y2 == y2);
        Piece targetPiece = state[x2, y2]; // 可能被吃的棋子
        if (targetPiece is Knight knight && knight.upgradeLevel == 1)
        {
            state.AddLianHuanMaTarget(knight.thisx, knight.thisy, knight.thisTeam);
        }

        if (!pieceMap.TryGetValue(selectedPiece, out GameObject pieceGO))
        {
            isMoving = false;
            yield break;
        }

        int x1 = selectedPiece.thisx, y1 = selectedPiece.thisy;

        // 检查是否为赛车的拐弯移动
        bool isSaiCheTurn = (selectedPiece.type == PieceType.Rook && selectedPiece.upgradeLevel == 1)
                            && (x1 != x2 && y1 != y2);

        // 定义路径检查局部函数（与规则层 IsBlocked 统一）
        bool IsPathClear(int fromX, int fromY, int toX, int toY)
        {
            if (fromX != toX && fromY != toY) return false;
            int dx = 0, dy = 0;
            if (fromX == toX) dy = (toY > fromY) ? 1 : -1;
            else dx = (toX > fromX) ? 1 : -1;
            int cx = fromX + dx, cy = fromY + dy;
            while (cx != toX || cy != toY)
            {
                if (state.IsBlocked(cx, cy, selectedPiece.thisTeam))   // 使用统一的阻挡判断
                    return false;
                cx += dx;
                cy += dy;
            }
            return true;
        }

        // 确定中间拐点（赛車且拐弯时）
        Vector3 midPoint = Vector3.zero;
        bool hasMidPoint = false;
        if (isSaiCheTurn)
        {
            // 两个候选拐点
            int mid1X = x1, mid1Y = y2; // 先纵后横
            int mid2X = x2, mid2Y = y1; // 先横后纵

            // 选择一条完全畅通的路径（拐点必须为空，且两段直线畅通）
            // 使用 IsBlocked 判断拐点是否可穿过（空格或己方墙）
            if (!state.IsBlocked(mid1X, mid1Y, selectedPiece.thisTeam) &&
                IsPathClear(x1, y1, mid1X, mid1Y) &&
                IsPathClear(mid1X, mid1Y, x2, y2))
            {
                midPoint = GetWorldPos(mid1X, mid1Y);
                hasMidPoint = true;
            }
            else if (!state.IsBlocked(mid2X, mid2Y, selectedPiece.thisTeam) &&
                     IsPathClear(x1, y1, mid2X, mid2Y) &&
                     IsPathClear(mid2X, mid2Y, x2, y2))
            {
                midPoint = GetWorldPos(mid2X, mid2Y);
                hasMidPoint = true;
            }
            // 根据游戏规则，至少有一条路径是合法的（IsLegalMove 已保证），无需 else
        }

        // 动画：放大到1.2倍
        Vector3 startScale = pieceGO.transform.localScale;
        Vector3 bigScale = Vector3.one * 1.2f;
        float elapsed = 0f;
        float zoomDuration = 0.1f;
        while (elapsed < zoomDuration)
        {
            pieceGO.transform.localScale = Vector3.Lerp(startScale, bigScale, elapsed / zoomDuration);
            elapsed += Time.deltaTime;
            yield return null;
        }
        pieceGO.transform.localScale = bigScale;

        // 移动动画
        Vector3 startPos = pieceGO.transform.position;
        Vector3 finalPos = GetWorldPos(x2, y2);

        if (hasMidPoint) // 赛車拐弯分两段
        {
            // 第一段：起点 -> 拐点
            yield return StartCoroutine(MoveBetween(pieceGO, startPos, midPoint, moveSpeed));
            // 第二段：拐点 -> 终点
            yield return StartCoroutine(MoveBetween(pieceGO, midPoint, finalPos, moveSpeed));
        }
        else // 普通直线移动
        {
            float distance = Vector3.Distance(startPos, finalPos);
            float moveDuration = distance / moveSpeed;
            elapsed = 0f;
            while (elapsed < moveDuration)
            {
                pieceGO.transform.position = Vector3.Lerp(startPos, finalPos, elapsed / moveDuration);
                elapsed += Time.deltaTime;
                yield return null;
            }
            pieceGO.transform.position = finalPos;
        }

        // 缩小回原始大小
        pieceGO.transform.localScale = Vector3.one;

        // 执行数据移动
        selectedPiece.Move(x2, y2, state);

        // 被吃棋子处理
        if (targetPiece.type != PieceType.Empty && targetPiece.isDead)
        {
            if (pieceMap.TryGetValue(targetPiece, out GameObject capturedGo))
                capturedGo.GetComponent<SpriteRenderer>().enabled = false;

            state.AddToGraveyard(targetPiece);
            yield return StartCoroutine(ProcessExplosions(targetPiece));
        }

        // 清理状态并切换回合
        DeselectPiece();
        if (state.lianHuanMaTeam == state.currentTeam && oldLianHuanMaTeam == state.currentTeam)
            state.ClearLianHuanMaTargets(state.currentTeam);
        EndTurnAndUpdate();
        isMoving = false;
    }
    // 辅助协程：匀速移动物体
    IEnumerator MoveBetween(GameObject obj, Vector3 from, Vector3 to, float speed)
    {
        float distance = Vector3.Distance(from, to);
        float duration = distance / speed;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            obj.transform.position = Vector3.Lerp(from, to, elapsed / duration);
            elapsed += Time.deltaTime;
            yield return null;
        }
        obj.transform.position = to;
    }
    IEnumerator SniperRoutine(SniperArrow arrow)
    {
        isMoving = true;
        // 移动后自动收起正在浏览的失子列表
        if (isShowingGraveyard)
            HideGraveyardList();

        int oldLianHuanMaTeam = state.lianHuanMaTeam;

        Pawn sniper = arrow.sniper;
        Piece target = sniper.GetSnipeTarget(arrow.dx, arrow.dy, state);
        if (target is Knight knight && knight.upgradeLevel == 1)
        {
            state.AddLianHuanMaTarget(knight.thisx, knight.thisy, knight.thisTeam);
        }
        // 无目标时不可能触发（箭头只在有目标时生成）
        if (target == null)
        {
            isMoving = false;
            yield break;
        }

        Vector3 sniperPos = pieceMap[sniper].transform.position;
        Vector3 targetPos = GetWorldPos(target.thisx, target.thisy);

        // 清除箭头和蓝点，取消选中
        DeselectPiece();

        // 启动后坐力动画（不等待，与射线并行）
        GameObject sniperGO = pieceMap[sniper];
        Coroutine recoilCoroutine = StartCoroutine(RecoilAnimation(sniperGO, sniperPos, arrow.dx, arrow.dy));

        // 创建临时射线
        GameObject beamObj = new GameObject("SniperBeam");
        LineRenderer lr = beamObj.AddComponent<LineRenderer>();
        lr.startWidth = 0.1f;
        lr.endWidth = 0.1f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = Color.red;
        lr.endColor = Color.red;
        lr.sortingOrder = 100;
        lr.positionCount = 2;
        lr.SetPosition(0, sniperPos);
        lr.SetPosition(1, sniperPos); // 起始点与终点重合

        // 射线飞行
        float distance = Vector3.Distance(sniperPos, targetPos);
        float flyTime = distance / 20f;
        float elapsed = 0f;
        while (elapsed < flyTime)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / flyTime);
            lr.SetPosition(1, Vector3.Lerp(sniperPos, targetPos, t));
            yield return null;
        }
        lr.SetPosition(1, targetPos);

        // 停留 0.3 秒
        yield return new WaitForSeconds(0.3f);

        // 确保后坐力动画结束
        yield return recoilCoroutine;

        // 销毁射线
        Destroy(beamObj);

        // 数据层狙击
        sniper.Snipe(arrow.dx, arrow.dy, state);

        // 被狙杀棋子视觉与坟墓处理
        if (target.isDead)
        {
            if (pieceMap.TryGetValue(target, out GameObject capturedGo))
                capturedGo.GetComponent<SpriteRenderer>().enabled = false;

            state.AddToGraveyard(target);
            yield return StartCoroutine(ProcessExplosions(target));
        }

        // 回合切换与冷却
        if (state.lianHuanMaTeam == state.currentTeam && oldLianHuanMaTeam == state.currentTeam)
            state.ClearLianHuanMaTargets(state.currentTeam);
        EndTurnAndUpdate();

        isMoving = false;
    }
    // 后坐力协程
    IEnumerator RecoilAnimation(GameObject obj, Vector3 originalPos, int dx, int dy)
    {
        Vector3 backward = new Vector3(-dx, -dy, 0).normalized * 0.1f;
        Vector3 recoilPos = originalPos + backward;
        float recoilDuration = 0.1f;
        float returnDuration = 0.2f;

        // 向后
        float elapsed = 0f;
        while (elapsed < recoilDuration)
        {
            obj.transform.position = Vector3.Lerp(originalPos, recoilPos, elapsed / recoilDuration);
            elapsed += Time.deltaTime;
            yield return null;
        }
        obj.transform.position = recoilPos;

        // 回归
        elapsed = 0f;
        while (elapsed < returnDuration)
        {
            obj.transform.position = Vector3.Lerp(recoilPos, originalPos, elapsed / returnDuration);
            elapsed += Time.deltaTime;
            yield return null;
        }
        obj.transform.position = originalPos;
    }
    // 处理自爆连锁
    IEnumerator ProcessExplosions(Piece deadPiece)
    {
        if (!(deadPiece is Pawn pawn) || !pawn.canExplode || pawn.upgradeLevel < 2)
            yield break;

        Vector3 center = pieceMap.TryGetValue(deadPiece, out GameObject deadGO)
            ? deadGO.transform.position
            : GetWorldPos(pawn.thisx, pawn.thisy);

        // 1. 播放粒子爆炸
        GameObject fx = null;
        if (explosionParticlePrefab != null)
        {
            fx = Instantiate(explosionParticlePrefab, center, Quaternion.identity);
            // 调整排序，让粒子在棋子上方
            var renderer = fx.GetComponent<ParticleSystemRenderer>();
            if (renderer != null) renderer.sortingOrder = 100;
        }

        // 2. 等待粒子生命周期结束（0.6秒）
        yield return new WaitForSeconds(0.6f);

        // 3. 清除特效
        if (fx != null) Destroy(fx);

        // 4. 执行周围击杀（数据层 + 视觉层）
        List<Piece> newDeads = pawn.Explode(state);
        foreach (Piece dead in newDeads)
        {
            if (pieceMap.TryGetValue(dead, out GameObject go))
                go.GetComponent<SpriteRenderer>().enabled = false;
            state.AddToGraveyard(dead);

            // 记录连环马目标（保持原有逻辑）
            if (dead is Knight knight && dead.upgradeLevel == 1)
                state.AddLianHuanMaTarget(knight.thisx, knight.thisy, knight.thisTeam);
        }

        // 5. 并发连锁爆炸（保持不变）
        List<Coroutine> concurrentExplosions = new List<Coroutine>();
        foreach (Piece dead in newDeads)
        {
            if (dead is Pawn deadPawn && deadPawn.canExplode && deadPawn.upgradeLevel >= 2)
                concurrentExplosions.Add(StartCoroutine(ProcessExplosions(dead)));
        }
        foreach (Coroutine coroutine in concurrentExplosions)
            yield return coroutine;
    }
    /// 在指定位置生成棋子并播放淡入动画（0.3秒透明度0→1）
    IEnumerator SpawnPieceAnimation(int x, int y, PieceType type)
    {
        // 1. 保留旧棋子引用（用于后续视觉隐藏和爆炸触发）
        Piece oldPiece = state[x, y];
        //收起信息版
        if (isShowingGraveyard)
            HideGraveyardList();

        // 2. 若旧棋子非空非墙，先禁用其渲染器并移除映射
        if (oldPiece.type != PieceType.Empty && oldPiece.type != PieceType.Wall)
        {
            if (pieceMap.TryGetValue(oldPiece, out GameObject oldGO))
            {
                oldGO.GetComponent<SpriteRenderer>().enabled = false;
            }
            pieceMap.Remove(oldPiece);
        }

        // 3. 数据层生成新棋子（内部会击杀旧棋子、入坟、清格）
        Piece newPiece = PieceGenerator.PlacePieceAt(state, x, y, type);
        if (newPiece == null) yield break;

        // 4. 视觉层创建新棋子游戏物体
        Vector3 worldPos = GetWorldPos(x, y);
        GameObject go = Instantiate(piecePrefab, worldPos, Quaternion.identity, transform);
        PieceView view = go.GetComponent<PieceView>();
        view.Setup(newPiece);

        SpriteRenderer sr = go.GetComponent<SpriteRenderer>();
        if (sr != null)
        {
            Color c = sr.color;
            c.a = 0f;
            sr.color = c;
        }
        pieceMap[newPiece] = go;

        // 5. 淡入动画（0.3秒）
        float duration = 0.3f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            if (sr != null)
            {
                Color c = sr.color;
                c.a = Mathf.Lerp(0f, 1f, elapsed / duration);
                sr.color = c;
            }
            yield return null;
        }
        if (sr != null)
        {
            Color c = sr.color;
            c.a = 1f;
            sr.color = c;
        }

        // 6. 若旧棋子被击杀且是自爆兵，触发爆炸链（协程独立运行，不等待）
        if (oldPiece.type != PieceType.Empty && oldPiece.type != PieceType.Wall && oldPiece.isDead)
        {
            StartCoroutine(ProcessExplosions(oldPiece));
        }
    }
    IEnumerator TongLeLaoWoAnimation(List<(int x, GameObject go)> newPawns)
    {
        isMoving = true;

        // 从左到右依次启动淡入（0.1秒间隔），每个淡入持续0.3秒
        List<Coroutine> fadeCoroutines = new List<Coroutine>();
        foreach (var (x, go) in newPawns)
        {
            Coroutine fade = StartCoroutine(FadeInPiece(go, 0.3f));
            fadeCoroutines.Add(fade);
            yield return new WaitForSeconds(0.1f); // 启动下一个前等待0.1秒
        }

        // 等待所有淡入完成
        foreach (Coroutine c in fadeCoroutines)
            yield return c;

        // 结束回合
        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator FadeInPiece(GameObject go, float duration)
    {
        SpriteRenderer sr = go.GetComponent<SpriteRenderer>();
        if (sr == null) yield break;
        Color transparent = new Color(1, 1, 1, 0);
        Color opaque = new Color(1, 1, 1, 1);
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            sr.color = Color.Lerp(transparent, opaque, elapsed / duration);
            yield return null;
        }
        sr.color = opaque;
    }
    IEnumerator SpawnWallAnimation()
    {
        isMoving = true;
        int team = state.currentTeam;
        int wx = 4;
        int wy = (team == 1) ? 1 : 8;

        // 如果中心已经是墙：重置倒计时，不播放动画
        if (state[wx, wy] is Wall existingWall)
        {
            existingWall.wallDuration = 10;
            if (pieceMap.TryGetValue(existingWall, out GameObject existingGO))
                existingGO.GetComponent<PieceView>().SetupWallCountdown(10);
            EndTurnAndUpdate();
            isMoving = false;
            yield break;
        }
        // 中心是己方棋子且不允许友伤：直接结束
        if (state[wx, wy].type != PieceType.Empty && state[wx, wy].thisTeam == team && Piece.friendlyFire != 1)
        {
            EndTurnAndUpdate();
            isMoving = false;
            yield break;
        }

        // 保存旧棋子引用
        Piece oldPiece = state[wx, wy];

        // 数据层生成墙（内部处理击杀）
        PieceGenerator.GenerateWall(state, team);

        // 隐藏旧棋子的渲染器（如果有）
        if (oldPiece.type != PieceType.Empty && oldPiece.type != PieceType.Wall)
        {
            if (pieceMap.TryGetValue(oldPiece, out GameObject oldGO))
                oldGO.GetComponent<SpriteRenderer>().enabled = false;
            pieceMap.Remove(oldPiece);
        }

        // 创建墙的视觉物体
        Vector3 worldPos = GetWorldPos(wx, wy);
        GameObject go = Instantiate(piecePrefab, worldPos, Quaternion.identity, transform);
        PieceView view = go.GetComponent<PieceView>();
        view.Setup(state[wx, wy]); // 墙已放置，直接Setup

        go.transform.localScale = Vector3.zero;
        pieceMap[state[wx, wy]] = go;

        // 弹出动画 0.3 秒
        float duration = 0.3f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            go.transform.localScale = Vector3.Lerp(Vector3.zero, Vector3.one, elapsed / duration);
            yield return null;
        }
        go.transform.localScale = Vector3.one;

        // 动画结束后才显示倒计角标
        if (state[wx, wy] is Wall newWall)
            view.SetupWallCountdown(newWall.wallDuration);

        // 处理被覆盖棋子的自爆链
        if (oldPiece.type != PieceType.Empty && oldPiece.type != PieceType.Wall && oldPiece.isDead)
        {
            StartCoroutine(ProcessExplosions(oldPiece));
        }

        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator DefectRoutine(Piece target, bool switchTurn = true)
    {
        isMoving = true;

        // 收起所有可能打开的面板（失子列表、信息面板等）
        if (isShowingGraveyard) HideGraveyardList();
        HideInfoPanel();

        if (!pieceMap.TryGetValue(target, out GameObject go))
        {
            isMoving = false;
            yield break;
        }

        SpriteRenderer sr = go.GetComponent<SpriteRenderer>();
        if (sr == null)
        {
            isMoving = false;
            yield break;
        }

        // 1. 色彩强度降至 0.3（0.2秒）
        float duration = 0.2f;
        float elapsed = 0f;
        Color startColor = new Color(1f, 1f, 1f, 1f);
        Color darkColor = new Color(0.3f, 0.3f, 0.3f, 1f);
        // 根据升级状态确定最终颜色（金色或白色）
        Color restoreColor = (target.upgradeLevel > 0) ? new Color(1f, 210f / 255f, 0f, 1f) : new Color(1f, 1f, 1f, 1f);
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            sr.color = Color.Lerp(startColor, darkColor, elapsed / duration);
            yield return null;
        }
        sr.color = darkColor;

        // 2. 执行叛变（数据层）
        target.Defect();

        // 3. 切换图片
        PieceView view = go.GetComponent<PieceView>();
        view.Setup(target);

        // 4. 色彩强度恢复至 1（0.2秒）
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            sr.color = Color.Lerp(darkColor, restoreColor, elapsed / duration);
            yield return null;
        }
        sr.color = restoreColor;

        // 5. 根据参数决定是否切换玩家
        if (switchTurn)
        {
            EndTurnAndUpdate();
        }

        isMoving = false;
    }
    /// 复活协程：执行数据复活、播放动画、处理自爆、切换回合
    IEnumerator ReviveRoutine(Piece piece, int x, int y)
    {
        isMoving = true;

        // 记录目标格原有棋子（用于后续处理爆炸）
        Piece targetPiece = state[x, y];

        // 1. 收起列表面板和蓝点
        ExitReviveMode();

        // 2. 视觉层：隐藏被覆盖的旧棋子
        if (targetPiece.type != PieceType.Empty && targetPiece.type != PieceType.Wall)
        {
            if (pieceMap.TryGetValue(targetPiece, out GameObject oldGO))
                oldGO.GetComponent<SpriteRenderer>().enabled = false;
            pieceMap.Remove(targetPiece);
        }

        // 3. 数据层：复活棋子
        PieceGenerator.RevivePiece(state, piece, x, y);
        piece.frozenTurns = 0;
        piece.freezeTickCount = 0;

        // 4. 重置狙击手冷却（若适用）
        if (piece is Pawn pawn && (pawn.upgradeLevel == 1 || pawn.upgradeLevel == 3))
        {
            pawn.sniperCooldown = 2;
            pawn.sniperAvailable = false;
        }

        // 5. 播放生成动画（淡入效果）- 直接复用已复活的棋子，不新建
        Vector3 reviveWorldPos = GetWorldPos(x, y);
        GameObject reviveGO = Instantiate(piecePrefab, reviveWorldPos, Quaternion.identity, transform);
        PieceView reviveView = reviveGO.GetComponent<PieceView>();
        reviveView.Setup(piece);
        reviveView.SetFrozen(false, piece.upgradeLevel > 0);
        SpriteRenderer reviveSR = reviveGO.GetComponent<SpriteRenderer>();
        if (reviveSR != null)
        {
            Color c = reviveSR.color;
            c.a = 0f;
            reviveSR.color = c;
        }
        pieceMap[piece] = reviveGO;

        float reviveDuration = 0.3f;
        float reviveElapsed = 0f;
        while (reviveElapsed < reviveDuration)
        {
            reviveElapsed += Time.deltaTime;
            if (reviveSR != null)
            {
                Color c2 = reviveSR.color;
                c2.a = Mathf.Lerp(0f, 1f, reviveElapsed / reviveDuration);
                reviveSR.color = c2;
            }
            yield return null;
        }
        if (reviveSR != null)
        {
            Color c3 = reviveSR.color;
            c3.a = 1f;
            reviveSR.color = c3;
        }

        // 6. 处理被击杀棋子引发的自爆链
        if (targetPiece.type != PieceType.Empty && targetPiece.type != PieceType.Wall && targetPiece.isDead)
        {
            StartCoroutine(ProcessExplosions(targetPiece));
        }

        // 7. 切换回合
        EndTurnAndUpdate();

        isMoving = false;
    }
    /// 反转视觉协程：所有棋子同时移动到对称位置并变色，持续 0.4 秒。
    IEnumerator ReverseAnimation(List<(Piece piece, GameObject go, Vector3 from, Vector3 to)> items)
    {
        isMoving = true;
        float duration = 0.4f;
        float halfDuration = 0.2f;
        Color darkColor = new Color(0.3f, 0.3f, 0.3f, 1f);
        bool imagesSwitched = false;

        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);

            foreach (var item in items)
            {
                // 位置插值
                item.go.transform.position = Vector3.Lerp(item.from, item.to, t);

                // 颜色控制
                SpriteRenderer sr = item.go.GetComponent<SpriteRenderer>();
                if (sr != null)
                {
                    // 根据棋子是否升级确定恢复颜色（升级等级在反转前后不变）
                    Color restoreColor = (item.piece.upgradeLevel > 0) ? new Color(1f, 210f / 255f, 0f, 1f) : Color.white;

                    if (elapsed < halfDuration)
                    {
                        // 前半段：从正常色变暗
                        float darkT = elapsed / halfDuration;
                        sr.color = Color.Lerp(restoreColor, darkColor, darkT);
                    }
                    else
                    {
                        // 第一次进入后半段时切换所有棋子图片
                        if (!imagesSwitched)
                        {
                            imagesSwitched = true;
                            foreach (var it in items)
                            {
                                it.go.GetComponent<PieceView>().Setup(it.piece);
                                // 强制设为暗色，避免闪白
                                SpriteRenderer sr2 = it.go.GetComponent<SpriteRenderer>();
                                if (sr2 != null) sr2.color = darkColor;
                            }
                        }
                        // 从暗色渐变回各自的恢复颜色
                        float brightT = (elapsed - halfDuration) / halfDuration;
                        sr.color = Color.Lerp(darkColor, restoreColor, brightT);
                    }
                }
            }
            yield return null;
        }

        // 确保最终状态：位置和颜色完全正确
        foreach (var item in items)
        {
            item.go.transform.position = item.to;
            SpriteRenderer sr = item.go.GetComponent<SpriteRenderer>();
            if (sr != null)
            {
                Color restoreColor = (item.piece.upgradeLevel > 0) ? new Color(1f, 210f / 255f, 0f, 1f) : Color.white;
                sr.color = restoreColor;
            }
        }

        // 如果没有进入后半段（duration 极小的情况），也确保图片切换
        if (!imagesSwitched)
        {
            foreach (var item in items)
                item.go.GetComponent<PieceView>().Setup(item.piece);
        }

        // 切换回合、冷却更新、保存状态
        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator FloodAnimation()
    {
        float leftEdge = -3f;
        float rightEdge = 11f;
        float yCenter = 4.5f;
        float barHeight = 2f;
        float totalWidth = rightEdge - leftEdge; // 14

        // 1. 创建洪水长条，pivot设在右边缘(1,0.5)，这样position=右边缘，scale.x向左增长
        GameObject floodBar = new GameObject("FloodBar");
        SpriteRenderer sr = floodBar.AddComponent<SpriteRenderer>();
        sr.sprite = Sprite.Create(CreateRectangleTexture(), new Rect(0, 0, 1, 1), new Vector2(1f, 0.5f), 1f);
        sr.color = new Color(0.667f, 0.667f, 1f, 1f);    // #AAAAFF
        sr.sortingOrder = 100;
        floodBar.transform.localScale = new Vector3(0f, barHeight, 1f);
        floodBar.transform.position = new Vector3(rightEdge, yCenter, 0f);

        // 2. 充入阶段
        float elapsed = 0f;
        while (elapsed < 0.2f)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / 0.2f);
            floodBar.transform.localScale = new Vector3(totalWidth * t, barHeight, 1f);
            yield return null;
        }
        floodBar.transform.localScale = new Vector3(totalWidth, barHeight, 1f);

        // 3. 停留 0.1 秒
        yield return new WaitForSeconds(0.1f);

        // 4. 消失阶段：左边缘固定，右边先消失
        sr.sprite = Sprite.Create(CreateRectangleTexture(), new Rect(0, 0, 1, 1), new Vector2(0f, 0.5f), 1f);
        floodBar.transform.position = new Vector3(leftEdge, yCenter, 0f);
        elapsed = 0f;
        while (elapsed < 0.2f)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / 0.2f);
            float currentWidth = totalWidth * (1f - t);
            floodBar.transform.localScale = new Vector3(currentWidth, barHeight, 1f);
            yield return null;
        }
        Destroy(floodBar);
    }
    /// <summary>生成一个纯白 1x1 Texture2D</summary>
    private Texture2D CreateRectangleTexture()
    {
        Texture2D tex = new Texture2D(1, 1);
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();
        return tex;
    }
    IEnumerator ChargeBugleAnimation(List<(GameObject go, Vector3 from, Vector3 to)> moves, List<GameObject> deadGOs)
    {
        isMoving = true;

        // 移动动画（0.2秒，同时进行）
        float duration = 0.2f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / duration;
            foreach (var (go, from, to) in moves)
                go.transform.position = Vector3.Lerp(from, to, t);
            yield return null;
        }
        // 确保到位
        foreach (var (go, from, to) in moves)
            go.transform.position = to;

        // 隐藏所有死亡棋子
        foreach (GameObject go in deadGOs)
        {
            // 获取对应的 Piece 引用（通过 pieceMap 反向查找或直接传递）
            // 这里由于我们只有 go，需要找到对应的 Piece 才能移除映射和触发自爆
            // 方法：从 pieceMap 中移除所有值等于 go 的条目
            Piece deadPiece = null;
            foreach (var kvp in pieceMap)
            {
                if (kvp.Value == go)
                {
                    deadPiece = kvp.Key;
                    break;
                }
            }
            if (deadPiece != null)
            {
                pieceMap.Remove(deadPiece);
                go.GetComponent<SpriteRenderer>().enabled = false;

                // 如果是自爆兵，触发爆炸
                if (deadPiece is Pawn pawn && pawn.canExplode)
                    StartCoroutine(ProcessExplosions(deadPiece));
            }
        }

        // 回合切换、冷却更新、保存状态
        EndTurnAndUpdate();
        isMoving = false;
    }
    /// 创建朝右的正三角形纹理（边长 = sideLength，宽 = 高*√3/2，高 = sideLength）
    /// 右顶点在右侧，pivot 应设为 (1, 0.5)
    Texture2D CreateRightTriangleTexture(float sideLength)
    {
        float height = sideLength;
        float width = height * Mathf.Sqrt(3f) / 2f;
        int texWidth = Mathf.CeilToInt(width * 100f);
        int texHeight = Mathf.CeilToInt(height * 100f);
        Texture2D tex = new Texture2D(texWidth, texHeight, TextureFormat.RGBA32, false);
        Color[] colors = new Color[texWidth * texHeight];
        float halfH = height / 2f;
        for (int y = 0; y < texHeight; y++)
        {
            // 当前行对应的世界坐标 v = (y + 0.5f) / texHeight * height (中心对齐)
            float worldY = ((y + 0.5f) / texHeight) * height - halfH; // 相对于中心线的偏移
            float distFromCenter = Mathf.Abs(worldY);
            // 允许的最大 x 值（从左到右逐渐收窄）
            float maxLocalX = (halfH - distFromCenter) / halfH * width; // 三角形内 x 上限
            for (int x = 0; x < texWidth; x++)
            {
                float localX = (x + 0.5f) / texWidth * width;
                if (localX <= maxLocalX)
                    colors[y * texWidth + x] = Color.white;
                else
                    colors[y * texWidth + x] = Color.clear;
            }
        }
        tex.SetPixels(colors);
        tex.Apply();
        return tex;
    }
    IEnumerator LaserAnimation(int targetY)
    {
        isMoving = true;
        float leftEdge = -3f;
        float rightEdge = 11f;
        float totalWidth = rightEdge - leftEdge; // 14

        // ----- 创建朝右的正三角形（边长1，右顶点为定位点）-----
        GameObject triangle = new GameObject("LaserTriangle");
        SpriteRenderer triSR = triangle.AddComponent<SpriteRenderer>();
        // 生成三角形纹理：宽0.866f，高1.0f
        Texture2D triTex = CreateRightTriangleTexture(1f); // 边长1的等边三角形
        triSR.sprite = Sprite.Create(triTex, new Rect(0, 0, triTex.width, triTex.height), new Vector2(1f, 0.5f), 100f);
        triSR.color = Color.red;
        triSR.sortingOrder = 100;
        // 初始随机行
        float startY = Random.Range(0, 10);
        triangle.transform.position = new Vector3(leftEdge, startY, 0f);

        // ----- 闪烁阶段：2秒内随机跳动 -----
        float flashDuration = 2.0f;
        float elapsed = 0f;
        // 初始等待0.05s后开始计算跳动
        yield return new WaitForSeconds(0.05f);
        elapsed += 0.05f;

        while (elapsed < flashDuration)
        {
            // 计算当前进度，用于间隔插值（从0.1s逐渐增长到0.4s）
            float progress = Mathf.Clamp01(elapsed / flashDuration);
            float wait = Mathf.Lerp(0.1f, 0.4f, progress);
            yield return new WaitForSeconds(wait);
            elapsed += wait;
            if (elapsed >= flashDuration) break;

            // 随机更换到任意整数行（0~9）
            int randY = Random.Range(0, 10);
            triangle.transform.position = new Vector3(leftEdge, randY, 0f);

            // 再停留0.05s
            yield return new WaitForSeconds(0.05f);
            elapsed += 0.05f;
        }

        // 最后移动到目标行，停留0.2秒
        triangle.transform.position = new Vector3(leftEdge, targetY, 0f);
        yield return new WaitForSeconds(0.2f);


        // ----- 抽出红色长条（从左向右，覆盖目标行）-----
        // 方法与洪水类似，使用 pivot 在左边缘，scale 向右伸展
        GameObject laserBar = new GameObject("LaserBar");
        SpriteRenderer barSR = laserBar.AddComponent<SpriteRenderer>();
        barSR.sprite = Sprite.Create(CreateRectangleTexture(), new Rect(0, 0, 1, 1), new Vector2(0f, 0.5f), 1f);
        barSR.color = new Color(0.933f, 0f, 0f, 1f); // #EE0000
        barSR.sortingOrder = 100;
        laserBar.transform.localScale = new Vector3(0f, 1f, 1f);    // 高度1单位，初始宽度为0
        laserBar.transform.position = new Vector3(leftEdge, targetY, 0f);

        // 向右抽出（0.2秒）
        float extractDuration = 0.2f;
        elapsed = 0f;
        while (elapsed < extractDuration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / extractDuration);
            laserBar.transform.localScale = new Vector3(totalWidth * t, 1f, 1f);
            yield return null;
        }
        laserBar.transform.localScale = new Vector3(totalWidth, 1f, 1f);

        // 停留0.1秒，清理死亡棋子视觉
        yield return new WaitForSeconds(0.1f);

        List<Piece> deadPieces = new List<Piece>();
        foreach (var kvp in pieceMap)
        {
            Piece piece = kvp.Key;
            if (piece.isDead && piece.thisy == targetY && piece.type != PieceType.Empty)
            {
                deadPieces.Add(piece);
                kvp.Value.GetComponent<SpriteRenderer>().enabled = false;
            }
        }
        foreach (var p in deadPieces)
            pieceMap.Remove(p);

        // 长条淡出（0.5秒）
        float fadeDuration = 0.5f;
        elapsed = 0f;
        Color originalColor = barSR.color;
        while (elapsed < fadeDuration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / fadeDuration;
            barSR.color = new Color(originalColor.r, originalColor.g, originalColor.b, 1f - t);
            yield return null;
        }
        Destroy(laserBar);
        Destroy(triangle);

        // ----- 处理被击杀棋子引发的自爆 -----
        foreach (Piece p in deadPieces)
        {
            if (p is Pawn pawn && pawn.canExplode)
                StartCoroutine(ProcessExplosions(p));
        }

        // 回合切换、冷却更新、保存状态
        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator BoardExpandCameraAnimation()
    {
        Camera cam = Camera.main;
        float startSize = cam.orthographicSize;
        float endSize = startSize + 1f;

        float elapsed = 0f;
        float duration = 0.5f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            cam.orthographicSize = Mathf.Lerp(startSize, endSize, t);
            yield return null;
        }
        cam.orthographicSize = endSize;
        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator BoardShrinkCameraAnimation()
    {
        Camera cam = Camera.main;
        float startSize = cam.orthographicSize;
        float endSize = startSize - 1f;

        float elapsed = 0f;
        float duration = 0.5f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            cam.orthographicSize = Mathf.Lerp(startSize, endSize, t);
            yield return null;
        }
        cam.orthographicSize = endSize;
        EndTurnAndUpdate();
        isMoving = false;
    }
    IEnumerator LotteryAnimation()
    {
        isMoving = true;
        lotteryButton.interactable = false;

        // 1. 滚动阶段：持续1秒，每0.05秒在按钮文本上随机显示一个名称
        TMP_Text btnText = lotteryButton.GetComponentInChildren<TMP_Text>();
        string originalText = btnText != null ? btnText.text : "";
        float rollDuration = 1.0f;
        float rollInterval = 0.05f;
        float rollEndTime = Time.time + rollDuration;
        while (Time.time < rollEndTime)
        {
            int randomIndex = Random.Range(1, 41); // 1~40
            if (btnText != null) btnText.text = prizeNames[randomIndex];
            yield return new WaitForSeconds(rollInterval);
        }

        // 2. 确定最终奖项
        int finalDraw = LotteryManager.Draw();
        string finalName = prizeNames[finalDraw];
        if (btnText != null) btnText.text = finalName;
        // 显示描述（未中奖和起死回生不显示）
        int[] noDisplay = { 17, 18, 19, 20, 21, 36, 37, 38, 39, 40 };
        if (System.Array.IndexOf(noDisplay, finalDraw) == -1)
        {
            lotteryEffectDiscription.Show(finalDraw);
        }

        // 3. 停留1秒
        yield return new WaitForSeconds(1.0f);

        // 4. 恢复交互和按钮文字
        isMoving = false;
        lotteryButton.interactable = true;
        if (btnText != null) btnText.text = originalText;

        // 5. 执行对应的抽奖效果
        switch (finalDraw)
        {
            case 1: EnterUpgradeMode(PieceType.Rook, 1); break;
            case 2: EnterUpgradeMode(PieceType.Cannon, 1); break;
            case 3: EnterUpgradeMode(PieceType.Cannon, 2); break;
            case 4: EnterUpgradeMode(PieceType.Pawn, 1); break;
            case 5: AutoUpgradeKing(); break;
            case 6: EnterUpgradeMode(PieceType.Bishop, 1); break;
            case 7: EnterUpgradeMode(PieceType.Bishop, 2); break;
            case 8: EnterUpgradeMode(PieceType.Pawn, 2); break;
            case 11: EnterUpgradeMode(PieceType.Knight, 1, 2); break;
            case 12: EnterUpgradeMode(PieceType.Guard, 1); break;
            case 16: EnterDefectMode(); break;
            case 28: case 29: case 30: EnterFreezeMode(); break;
            case 31: case 32: case 33: case 34: case 35: EnterDefrostMode(); break;
            case 13: ApplyGeneration(PieceType.Cannon, 2); break;
            case 14: ApplyGeneration(PieceType.Knight, 2); break;
            case 15: ApplyGeneration(PieceType.Rook, 1); break;
            case 17: case 18: case 19: case 20: case 21: EnterReviveMode(); break;
            case 9: ApplyTongLeLaoWo(); break;
            case 10: ApplyWall(); break;
            case 22: ApplyReverse(); break;
            case 23: ApplyFlood(); break;
            case 24: ApplyChargeBugle(); break;
            case 25: ApplyLaserCannon(); break;
            case 26: ApplyBoardExpand(); break;
            case 27: ApplyBoardShrink(); break;
            default: EndTurnAndUpdate(); break; // 未中奖直接结束回合
        }
    }
    //-----------------------------------

    public void OnCellClicked(int x, int y)
    {
        if (isMoving) return;   // 动画播放中，忽略点击
        if (!aiActing && gameMode == GameMode.AI && state.currentTeam == aiTeam) return;

        // 延迟隐藏抽奖描述：回合切换后第一次点击时消失
        if (pendingHideDescription)
        {
            pendingHideDescription = false;
            if (lotteryEffectDiscription != null)
                lotteryEffectDiscription.Hide();
        }

        Piece clickedPiece = state[x, y];
        // 冻结模式拦截
        if (isFreezeMode)
        {
            Piece clicked = state[x, y];
            if (clicked != null && freezeTargets.Contains(clicked))
            {
                // 执行冻结
                clicked.frozenTurns = 6;
                clicked.freezeTickCount = 0;
                if (pieceMap.TryGetValue(clicked, out GameObject go))
                    go.GetComponent<PieceView>().SetFrozen(true, clicked.upgradeLevel > 0);

                // 退出冻结模式并切换回合
                isFreezeMode = false;
                freezeTargets.Clear();
                ClearDots();
                EndTurnAndUpdate();
            }
            // 点击其他格子不做任何事
            return;
        }
        // 解冻模式拦截
        if (isDefrostMode)
        {
            Piece clicked = state[x, y];
            if (clicked != null && defrostTargets.Contains(clicked))
            {
                // 解冻棋子
                clicked.frozenTurns = 0;
                clicked.freezeTickCount = 0;
                if (pieceMap.TryGetValue(clicked, out GameObject go))
                    go.GetComponent<PieceView>().SetFrozen(false, clicked.upgradeLevel > 0);

                // 退出解冻模式，进入移动子状态
                isDefrostMode = false;
                defrostTargets.Clear();
                ClearDots();

                isPostDefrostMove = true;
                defrostedPiece = clicked;
                SelectPiece(clicked);

                // 在自身位置额外画一个蓝点（表示可选择原地不动）
                Vector3 selfPos = GetWorldPos(clicked.thisx, clicked.thisy);
                GameObject selfDot = Instantiate(blueDotPrefab, selfPos, Quaternion.identity, transform);
                activeDots.Add(selfDot);
                // 向合法移动列表中添加自身位置
                currentLegalMoves.Add(new LegalMove(clicked.thisx, clicked.thisy, clicked.thisx, clicked.thisy));
            }
            // 点击其他格子忽略
            return;
        }
        // 解冻后强制移动子状态
        if (isPostDefrostMove)
        {
            // 点击自身位置 → 放弃移动，直接结束回合
            if (x == defrostedPiece.thisx && y == defrostedPiece.thisy)
            {
                isPostDefrostMove = false;
                defrostedPiece = null;
                DeselectPiece();
                EndTurnAndUpdate();
                return;
            }

            // 点击合法移动目标 → 正常移动（移动协程内部会调用EndTurnAndUpdate）
            if (currentLegalMoves != null)
            {
                foreach (LegalMove move in currentLegalMoves)
                {
                    if (move.x2 == x && move.y2 == y)
                    {
                        // 确认不是原地不动（原地不动已在上面处理）
                        if (move.x1 != move.x2 || move.y1 != move.y2)
                        {
                            isPostDefrostMove = false;
                            defrostedPiece = null;
                            StartCoroutine(MoveToTarget(x, y));
                            return;
                        }
                    }
                }
            }
            // 点击其他格子忽略
            return;
        }
        // 升级模式拦截
        if (isUpgradeMode)
        {
            if (clickedPiece != null && upgradeTargets.Contains(clickedPiece))
            {
                // 执行升级
                int newLevel = clickedPiece.upgradeLevel + currentUpgradeEffectLevel;
                clickedPiece.Upgrade(newLevel);
                if (pieceMap.TryGetValue(clickedPiece, out GameObject upGO))
                    upGO.GetComponent<PieceView>().SetUpgraded(clickedPiece.upgradeLevel);

                upgradesRemaining--;
                upgradeTargets.Remove(clickedPiece);

                if (upgradesRemaining <= 0 || upgradeTargets.Count == 0)
                {
                    ExitUpgradeMode();
                    EndTurnAndUpdate();
                }
                else
                {
                    // 刷新蓝点
                    ClearDots();
                    foreach (Piece p in upgradeTargets)
                    {
                        Vector3 pos = GetWorldPos(p.thisx, p.thisy);
                        GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
                        activeDots.Add(dot);
                    }
                }
            }
            return;
        }
        // 复活模式拦截
        if (isReviveMode)
        {
            // 如果已选中棋子且点击了合法复活位置
            if (selectedRevivePiece != null)
            {
                for (int i = 0; i < currentRevivePositions.Count; i++)
                {
                    if (currentRevivePositions[i].x == x && currentRevivePositions[i].y == y)
                    {
                        // 执行复活
                        StartCoroutine(ReviveRoutine(selectedRevivePiece, x, y));
                        return;
                    }
                }
            }
            // 点击其他位置不做任何事（不可主动放弃）
            return;
        }
        // 叛变模式拦截
        if (isDefectMode)
        {
            Piece clicked = state[x, y];
            if (clicked != null && defectTargets.Contains(clicked))
            {
                ExitDefectMode();
                StartCoroutine(DefectRoutine(clicked));
            }
            return;
        }
        // 生成模式拦截
        if (isGenerationMode)
        {
            if (PieceGenerator.IsPositionValidForGeneration(state, x, y, currentGenerationType, state.currentTeam))
            {
                // 执行生成
                StartCoroutine(SpawnPieceAnimation(x, y, currentGenerationType));
                generationRemaining--;

                // 检查是否完成生成或没有合法位置
                if (generationRemaining <= 0)
                {
                    ExitGenerationMode(true);
                }
                else
                {
                    // 刷新蓝点（棋盘状态已变，重新扫描合法位置）
                    ClearDots();
                    int team = state.currentTeam;
                    bool anyValid = false;
                    for (int px = state.leftBound; px <= state.rightBound; px++)
                    {
                        for (int py = state.lowerBound; py <= state.upperBound; py++)
                        {
                            if (PieceGenerator.IsPositionValidForGeneration(state, px, py, currentGenerationType, team))
                            {
                                Vector3 pos = GetWorldPos(px, py);
                                GameObject dot = Instantiate(blueDotPrefab, pos, Quaternion.identity, transform);
                                activeDots.Add(dot);
                                anyValid = true;
                            }
                        }
                    }
                    if (!anyValid)
                    {
                        // 没有合法位置，提前结束
                        ExitGenerationMode(true);
                    }
                }
            }
            return;
        }
        // 准备阶段：不允许直接选中棋子移动，只能通过抽奖行动
        if (state.prepareModeOn)
        {
            return;
        }
        // 没有棋子被选中
        if (selectedPiece == null)
        {
            // 点击己方棋子 -> 选中
            if (clickedPiece.type != PieceType.Empty && clickedPiece.thisTeam == state.currentTeam)
            {
                SelectPiece(clickedPiece);
            }
        }
        // 已有棋子被选中
        else
        {
            // 检查是否点击了蓝点（合法目标）
            bool isLegalTarget = false;
            if (currentLegalMoves != null)
            {
                foreach (LegalMove move in currentLegalMoves)
                {
                    if (move.x2 == x && move.y2 == y)
                    {
                        isLegalTarget = true;
                        break;
                    }
                }
            }

            if (isLegalTarget)
            {
                StartCoroutine(MoveToTarget(x, y));
            }
            else
            {
                //点击已经选中的棋子 -> 取消选中
                if (clickedPiece == selectedPiece)
                {
                    DeselectPiece();
                }
                // 点击的是己方其他棋子 -> 切换选中
                else if (clickedPiece.type != PieceType.Empty && clickedPiece.thisTeam == state.currentTeam)
                {
                    DeselectPiece();
                    SelectPiece(clickedPiece);
                }
                // 其他情况（空格、敌方棋子） -> 取消选中
                else
                {
                    DeselectPiece();
                }
            }
        }
    }
    /// <summary> 由 SniperArrow 调用 </summary>
    public void HideInfoPanel()
    {
        if (infoPanelInstance != null)
            infoPanelInstance.SetActive(false);
    }
    public void OnCellLongPress(int x, int y)
    {
        Piece piece = state[x, y];
        if (piece == null || piece.type == PieceType.Empty) return;

        if (infoPanelInstance != null && infoText != null)
        {
            RectTransform rt = infoPanelInstance.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(3f, 1f);

            // 将棋子世界坐标转换为Canvas本地坐标
            Vector3 worldPos = GetWorldPos(x, y) + Vector3.up * 0.8f;
            Camera cam = Camera.main;
            Canvas canvas = FindObjectOfType<Canvas>();
            if (cam != null && canvas != null)
            {
                Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(cam, worldPos);
                RectTransformUtility.ScreenPointToLocalPointInRectangle(
                    canvas.GetComponent<RectTransform>(), screenPos, cam, out Vector2 localPos);
                rt.localPosition = localPos;
            }

            infoText.text = piece.GetUpgradeName();
            infoText.fontSize = 0.5f;
            infoText.alignment = TextAlignmentOptions.Center;
            infoPanelInstance.SetActive(true);
        }
    }
    public void OnSniperArrowClicked(SniperArrow arrow)
    {
        if (isMoving) return;   // 动画播放中，忽略点击
        if (!aiActing && gameMode == GameMode.AI && state.currentTeam == aiTeam) return;
        if (arrow.sniper.frozenTurns > 0) return;
        StartCoroutine(SniperRoutine(arrow));
    }
    public void UndoMove()
    {
        if (isMoving) return; // 动画播放中不允许悔棋，直接忽略
        if (isReviveMode) return; // 复活模式中不允许悔棋
        if (isUpgradeMode) return; // 升级模式中不允许悔棋
        if (isDefectMode) return; // 叛变模式中不允许悔棋
        if (isGenerationMode) return; // 生成模式中不允许悔棋
        if (isFreezeMode) return; // 冻结模式中不允许悔棋
        if (isDefrostMode) return; // 解冻模式中不允许悔棋
        if (isPostDefrostMove) return; // 解冻后强制移动模式中不允许悔棋

        if (stateHistory.Count < 2) return;

        stateHistory.RemoveAt(0);
        state = stateHistory[0].DeepClone();

        DeselectPiece();
        // 恢复行动方提示
        UpdateTurnIndicator();
        // 此时没有动画，无需 StopAllCoroutines
        // isMoving 已经为 false

        // 恢复相机到悔棋前的棋盘范围
        Camera cam = Camera.main;
        float targetSize = state.isBoardExpanded ? 7.5f : 6.5f;
        if (cam != null && !Mathf.Approximately(cam.orthographicSize, targetSize))
        {
            cam.orthographicSize = targetSize;
        }

        foreach (GameObject go in pieceMap.Values)
            Destroy(go);
        pieceMap.Clear();

        GenerateAllPieces();
    }

    //------------------------------------
    //--------更多UI管理-------------------
    //-------------------------------------
    /// <summary>点击“显示红方失子”按钮</summary>
    public void OnShowRedGraveyard()
    {
        if (isReviveMode) return; // 复活模式中不响应

        if (isShowingGraveyard && showingGraveyardTeam == 1)
        {
            // 当前正在展示红方，点击则收起
            HideGraveyardList();
        }
        else
        {
            // 展示红方坟墓列表
            ShowGraveyardList(1);
        }
    }
    /// <summary>点击“显示黑方失子”按钮</summary>
    public void OnShowBlackGraveyard()
    {
        if (isReviveMode) return;

        if (isShowingGraveyard && showingGraveyardTeam == -1)
        {
            HideGraveyardList();
        }
        else
        {
            ShowGraveyardList(-1);
        }
    }
    /// <summary>展示指定阵营的坟墓列表（仅浏览，不可点击）</summary>
    private void ShowGraveyardList(int team)
    {
        // 先清除之前的列表
        ClearReviveListItems();
        HideGraveyardList();

        isShowingGraveyard = true;
        showingGraveyardTeam = team;

        // 生成列表项（不可点击）
        List<Piece> graveyard = (team == 1) ? state.redGraveyard : state.blackGraveyard;
        Transform content = reviveListPanel.transform.Find("Viewport/Content");
        foreach (Piece piece in graveyard)
        {
            GameObject item = Instantiate(reviveListItemPrefab, content);
            ReviveListItem listItem = item.GetComponent<ReviveListItem>();
            listItem.Setup(piece, this, false); // 不可点击
            reviveListItems.Add(item);
        }

        reviveListPanel.SetActive(true);
        buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);
    }
    /// <summary>隐藏浏览列表</summary>
    private void HideGraveyardList()
    {
        isShowingGraveyard = false;
        showingGraveyardTeam = 0;
        ClearReviveListItems();
        reviveListPanel.SetActive(false);
        buttonControl.UpdateGraveyardButtons(isShowingGraveyard, showingGraveyardTeam, isReviveMode);
    }
    /// <summary>清除所有列表项</summary>
    private void ClearReviveListItems()
    {
        foreach (GameObject item in reviveListItems)
        {
            Destroy(item);
        }
        reviveListItems.Clear();
    }
    /// 根据棋子类型和阵营获取对应的 Sprite。
    public Sprite GetSpriteForPiece(Piece piece)
    {
        if (pieceSprites == null || pieceSprites.Count < 14) return null;

        // 计算基础索引：红方 0-6，黑方 7-13，与 PieceView 顺序完全一致
        int baseIndex = ((int)piece.type - 1); // Rook=1 -> 0, Knight=2 -> 1, Bishop=3 -> 2, Guard=4 -> 3, King=5 -> 4, Cannon=6 -> 5, Pawn=7 -> 6
        int index = (piece.thisTeam == 1) ? baseIndex : baseIndex + 7;

        if (index < 0 || index >= pieceSprites.Count) return null;
        return pieceSprites[index];
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();
        // z键测试秦王绕柱（保留）
        if (Input.GetKeyDown(KeyCode.Z))
        {
            if (!isMoving)
            {
                ApplyWall();
            }
        }
        //L键领域展开/收缩
        if (Input.GetKeyDown(KeyCode.L))
        {
            if (!isMoving)
            {
                if (state.isBoardExpanded)
                    ApplyBoardShrink();
                else
                    ApplyBoardExpand();
            }
        }

        // ===== 测试代码开始：1/2炮升级，5将升级（可整块删除）=====
        if (!isMoving && !IsInInteractiveMode())
        {
            if (Input.GetKeyDown(KeyCode.Alpha1))
                EnterUpgradeMode(PieceType.Cannon, 1);
            else if (Input.GetKeyDown(KeyCode.Alpha2))
                EnterUpgradeMode(PieceType.Cannon, 2);
            else if (Input.GetKeyDown(KeyCode.Alpha5))
                AutoUpgradeKing();
        }
        // ===== 测试代码结束 =====

        // ===== 测试代码开始：空格切换模型/纯MCTS（可整块删除）=====
        if (Input.GetKeyDown(KeyCode.Space) && gameMode == GameMode.AI && aiController != null && !isMoving)
        {
            bool success = aiController.ToggleMode(aiTeam);
            StartCoroutine(FlashScreen(success));
        }
        // ===== 测试代码结束 =====

        // AI 回合
        if (gameMode == GameMode.AI && aiController != null &&
            state.currentTeam == aiTeam && !isMoving && !aiProcessing)
        {
            aiProcessing = true;
            StartCoroutine(AITurnCoroutine());
        }
    }

    // ===== 测试代码开始：闪屏协程（可整块删除）=====
    IEnumerator FlashScreen(bool white)
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null) yield break;

        GameObject flashGO = new GameObject("TestFlash");
        flashGO.transform.SetParent(canvas.transform, false);
        UnityEngine.UI.Image flashImg = flashGO.AddComponent<UnityEngine.UI.Image>();
        flashImg.color = new Color(1, 1, 1, 0);
        flashImg.rectTransform.anchorMin = Vector2.zero;
        flashImg.rectTransform.anchorMax = Vector2.one;
        flashImg.rectTransform.sizeDelta = Vector2.zero;
        flashImg.raycastTarget = false;

        Color targetColor = white ? Color.white : Color.black;
        float duration = 0.1f;
        float half = duration * 0.5f;

        // 淡入
        float elapsed = 0f;
        while (elapsed < half)
        {
            elapsed += Time.deltaTime;
            flashImg.color = Color.Lerp(new Color(targetColor.r, targetColor.g, targetColor.b, 0), targetColor, elapsed / half);
            yield return null;
        }
        // 淡出
        elapsed = 0f;
        while (elapsed < half)
        {
            elapsed += Time.deltaTime;
            flashImg.color = Color.Lerp(targetColor, new Color(targetColor.r, targetColor.g, targetColor.b, 0), elapsed / half);
            yield return null;
        }

        Destroy(flashGO);
    }
    // ===== 测试代码结束 =====

    //------------------------------------
    private bool IsInInteractiveMode()
    {
        return isUpgradeMode || isDefectMode || isFreezeMode || isDefrostMode ||
               isPostDefrostMove || isReviveMode || isGenerationMode;
    }

    private void AutoHandleInteractiveMode()
    {
        if (isUpgradeMode && upgradeTargets.Count > 0)
        {
            Piece target = upgradeTargets[Random.Range(0, upgradeTargets.Count)];
            OnCellClicked(target.thisx, target.thisy);
        }
        else if (isDefectMode && defectTargets.Count > 0)
        {
            var targetList = new List<Piece>(defectTargets);
            Piece target = targetList[Random.Range(0, targetList.Count)];
            OnCellClicked(target.thisx, target.thisy);
        }
        else if (isFreezeMode && freezeTargets.Count > 0)
        {
            Piece target = freezeTargets[Random.Range(0, freezeTargets.Count)];
            OnCellClicked(target.thisx, target.thisy);
        }
        else if (isDefrostMode && defrostTargets.Count > 0)
        {
            Piece target = defrostTargets[Random.Range(0, defrostTargets.Count)];
            OnCellClicked(target.thisx, target.thisy);
        }
        else if (isPostDefrostMove && currentLegalMoves != null && currentLegalMoves.Count > 0)
        {
            LegalMove move = currentLegalMoves[0];
            OnCellClicked(move.x2, move.y2);
        }
        else if (isReviveMode)
        {
            AutoRevive();
        }
        else if (isGenerationMode)
        {
            AutoGenerate();
        }
    }

    private void AutoRevive()
    {
        int team = state.currentTeam;
        List<Piece> graveyard = (team == 1) ? state.redGraveyard : state.blackGraveyard;

        foreach (Piece piece in graveyard)
        {
            var positions = state.GetInitialPositions(piece.type, piece.thisTeam);
            foreach (var (x, y) in positions)
            {
                if (IsPositionValidForRevive(x, y, piece.thisTeam))
                {
                    OnRevivePieceSelected(piece);
                    OnCellClicked(x, y);
                    return;
                }
            }
        }
    }

    private void AutoGenerate()
    {
        int team = state.currentTeam;
        for (int x = state.leftBound; x <= state.rightBound; x++)
        {
            for (int y = state.lowerBound; y <= state.upperBound; y++)
            {
                if (PieceGenerator.IsPositionValidForGeneration(state, x, y, currentGenerationType, team))
                {
                    OnCellClicked(x, y);
                    return;
                }
            }
        }
        ExitGenerationMode(true);
    }

    public void ExecuteAIAction(GameAction action)
    {
        if (action is MoveAction move)
        {
            selectedPiece = state[move.fromX, move.fromY];
            currentLegalMoves = new List<LegalMove> { new LegalMove(move.fromX, move.fromY, move.toX, move.toY) };
            StartCoroutine(MoveToTarget(move.toX, move.toY));
        }
        else if (action is SniperAction sniper)
        {
            GameObject tempObj = new GameObject("TempSniperArrow");
            SniperArrow arrow = tempObj.AddComponent<SniperArrow>();
            arrow.dx = sniper.dx;
            arrow.dy = sniper.dy;
            arrow.sniper = (Pawn)state[sniper.fromX, sniper.fromY];
            arrow.manager = this;
            StartCoroutine(AISniperCleanup(arrow, tempObj));
        }
        else if (action is LotteryAction)
        {
            OnLottery();
        }
    }

    IEnumerator AISniperCleanup(SniperArrow arrow, GameObject tempObj)
    {
        yield return StartCoroutine(SniperRoutine(arrow));
        if (tempObj != null) Destroy(tempObj);
    }

    IEnumerator AITurnCoroutine()
    {
        aiActing = true;
        yield return StartCoroutine(AITurnInner());
        aiActing = false;
    }

    IEnumerator AITurnInner()
    {
        // 思考延迟
        yield return new WaitForSeconds(0.5f);

        // 处理可能残留的交互模式
        while (IsInInteractiveMode())
        {
            AutoHandleInteractiveMode();
            if (isMoving)
                yield return new WaitWhile(() => isMoving);
            else
                yield return null;
            yield return new WaitForSeconds(0.2f);
        }

        // 回合可能已切换
        if (state.currentTeam != aiTeam)
        {
            aiProcessing = false;
            yield break;
        }

        // 正常 AI 行动
        GameAction action = aiController.GetBestAction(state);
        ExecuteAIAction(action);

        yield return new WaitWhile(() => isMoving);

        // 抽奖可能触发交互模式，循环处理
        while (IsInInteractiveMode())
        {
            AutoHandleInteractiveMode();
            if (isMoving)
                yield return new WaitWhile(() => isMoving);
            else
                yield return null;
            yield return new WaitForSeconds(0.2f);
        }

        aiProcessing = false;
    }
}