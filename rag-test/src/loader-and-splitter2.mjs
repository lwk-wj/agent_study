import "dotenv/config"
import "cheerio"
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"


// 初始化 ChatOpenAI语言模型
const model = new ChatOpenAI({
    temperature: 0,
    model: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL
    }
})

// 初始化 OpenAI 文本嵌入模型 (将文本转为高纬度向量，进行语义相似度计算)
const embedding = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: process.env.EMBEDDINGS_MODEL_NAME,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL
    }
})

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
console.log(`文档分割完成，共${splitDocuments.length}个分块\n`);

console.log("正在创建向量存储，请稍等...");

// 将文档注入内存向量数据库，使用embedding模型对文档内容进行向量化
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, embedding);

console.log("向量存储创建完成！\n");

const retriever = vectorStore.asRetriever({ k: 2 })


const questions = [
    "作者的人生态度产生了怎样的根本性逆转？"
]

for (const q of questions) {
    console.log("=".repeat(80));
    console.log(`\n问题：${q}`);
    console.log("=".repeat(80));

    // 检索最相关的文档分块
    const retrievedDocs = await retriever.invoke(q);

    const scoredResults = await vectorStore.similaritySearchWithScore(q, 2);

    console.log("\n【检索到的文档及相似度评分】");

    retrievedDocs.forEach((doc, i) => {
        // 找到对应的评分
        const scoredResult = scoredResults.find(([scoredDoc]) =>
            scoredDoc.pageContent === doc.pageContent
        );

        const score = scoredResult ? scoredResult[1] : null;
        const similarity = score !== null ? (1 - score).toFixed(4) : "N/A";
        console.log(`\n[文档 ${i + 1}] 相似度: ${similarity}`);
        console.log(`内容: ${doc.pageContent}`);
        if (doc.metadata && Object.keys(doc.metadata).length > 0) {
            console.log(`元数据:`, doc.metadata);
        }
    });

    const context = retrievedDocs
        .map((doc, i) => `[片段${i + 1}]\n${doc.pageContent}`)
        .join("\n\n━━━━━\n\n");



    const prompt = `你是一个文章辅助阅读助手，根据文章内容来解答：
                    文章内容：
                    ${context}
                    问题: ${q}
                    你的回答:`;
    console.log("\n【AI 回答】");
    const response = await model.invoke(prompt);
    console.log(response.content);
    console.log("\n");
}
