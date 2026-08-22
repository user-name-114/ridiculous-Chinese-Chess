// LotteryManager.cs
using System.Collections.Generic;

//纯数据文件，不含unity相关内容
public static class LotteryManager
{
    private static System.Random rng = new System.Random();

    /// <summary>随机抽取 1~40 的效果编号</summary>
    public static int Draw()
    {
        return rng.Next(1, 41); // 1 到 40 均匀分布
    }
}