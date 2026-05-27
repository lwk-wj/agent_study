import "dotenv/config"
//  用于网页解析的轻量级 jQuery 实现（LangChain 内部会依赖它）
import "cheerio"
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// 1. 初始化网页加载器（Web Loader）
// 使用 Cheerio 爬取指定的掘金文章网页，并通过 CSS 选择器筛选出正文段落（'.main-area p'）
const cheerioLoader = new CheerioWebBaseLoader("https://juejin.cn/post/7233327509919547452", {
    selector: ".main-area p",
})

// 2. 加载网页内容，返回包含网页纯文本和元数据（如 source 来源 url）的 Document 数组
const documents = await cheerioLoader.load()

// 3. 初始化递归字符文本分割器（RecursiveCharacterTextSplitter）
// 它的工作原理是按照优先级顺序（如段落、句子、词）逐步分割，直到每个分块（chunk）的大小符合要求
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 400,  // 每个分块的最大字符数限制
    chunkOverlap: 50,  // 相邻两个分块之间的重叠字符数，用于保留段落之间的语义上下文连续性
    separators: ["。", "！", "？"],  // 分割符列表，会按顺序尝试使用分割符分割文本
})

// 4. 将加载出来的完整网页文档（通常很大，一个网页通常是一个大文档）拆分成多个小文档分块
const splitDocuments = await textSplitter.splitDocuments(documents);

// 5. 打印分割后的小文档数组（每个分块包含部分文本内容和 metadata 数据）
console.log(splitDocuments);
