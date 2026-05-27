import "dotenv/config"; // 加载 .env 文件中的环境变量（如 OpenAI API Key、Base URL 等）
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node'; // 引入 Milvus 官方 Node.js SDK 及核心配置枚举
import { OpenAIEmbeddings } from "@langchain/openai"; // 引入 LangChain 提供的 OpenAI 向量生成工具

const COLLECTION_NAME = 'ai_diary'; // 设置集合（类似于关系数据库的表）的名称
const VECTOR_DIM = 1024; // 定义向量维度，这里使用的 Embedding 模型会输出 1024 维度的向量

// 初始化 LangChain 的 OpenAI Embeddings 客户端，用于将文本转换为向量
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY, // 从环境变量获取 API Key
    model: process.env.EMBEDDINGS_MODEL_NAME, // 使用的 Embedding 模型名称（如 text-embedding-3-large 等）
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL // 自定义 API 的基础请求地址（如使用国内代理或中转服务）
    },
    dimensions: VECTOR_DIM // 显式设定输出向量的维度大小
});

// 初始化 Milvus 客户端并指定服务器的连接地址和端口
const client = new MilvusClient({
    address: 'localhost:19530' // 默认本地 Milvus 端口为 19530
});

/**
 * 辅助函数：将一段文本内容传入 Embedding 模型，计算并返回其对应的稠密向量
 * @param {string} text - 需要进行向量化的日记内容
 * @returns {Promise<number[]>} - 返回代表该文本的 1024 维度的浮点数数组
 */
async function getEmbedding(text) {
    const result = await embeddings.embedQuery(text);
    return result;
}

async function main() {
    try {
        console.log('Connecting to Milvus...');
        // 等待客户端成功连接到 Milvus 数据库
        await client.connectPromise;
        console.log('✓ Connected\n');

        // 1. 创建集合（Collection）
        // 集合类似于传统关系型数据库中的“表”。在 Milvus 中，每个集合需要定义明确的 schema (字段结构)。
        console.log('Creating collection...');
        await client.createCollection({
            collection_name: COLLECTION_NAME,
            fields: [
                // id 字段：作为主键（is_primary_key: true），类型为可变长度字符串，最大长度 50
                { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
                // vector 字段：存放生成的文本向量，类型为浮点数向量（FloatVector），并指定必须为 1024 维度
                { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },
                // content 字段：存放日记的具体文字内容，最长支持 5000 字符
                { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
                // date 字段：存放日记日期
                { name: 'date', data_type: DataType.VarChar, max_length: 50 },
                // mood 字段：存放当天心情的分类标签（例如 happy, relaxed 等）
                { name: 'mood', data_type: DataType.VarChar, max_length: 50 },
                // tags 字段：存放日记标签数组，最高容量 10 个元素，每个元素最长 50 字符
                { name: 'tags', data_type: DataType.Array, element_type: DataType.VarChar, max_capacity: 10, max_length: 50 }
            ]
        });
        console.log('Collection created');

        // 2. 创建向量索引（Index）
        // 在向量字段上必须建立索引，否则后续无法进行高效的近似最近邻（ANN）相似度检索。
        console.log('\nCreating index...');
        await client.createIndex({
            collection_name: COLLECTION_NAME,
            field_name: 'vector', // 指定要建立索引的向量字段
            // IVF_FLAT：倒排文件平铺索引。它将向量空间划分为多个聚类网格（Voronoi Cells），
            // 检索时只查找与目标向量最近的几个聚类，极大加快了检索速度。
            index_type: IndexType.IVF_FLAT,
            // COSINE：余弦相似度度量标准。衡量两个向量方向的夹角，夹角越小（值越接近1）说明语义越相似，非常适合文本搜索。
            metric_type: MetricType.COSINE,
            // params.nlist：聚类中心的数量。这里设为 1024，即把所有向量空间聚类成 1024 个桶。
            params: { nlist: 1024 }
        });
        console.log('Index created');

        // 3. 加载集合（Load Collection）
        // 这一步是 Milvus 的核心特性：Milvus 采用内存中检索设计。
        // 在进行数据插入或查询前，必须显式地把集合加载到内存（RAM）中以保证检索的实时高速性能。
        console.log('\nLoading collection...');
        await client.loadCollection({ collection_name: COLLECTION_NAME });
        console.log('Collection loaded');

        // 4. 准备模拟的日记数据
        console.log('\nInserting diary entries...');
        const diaryContents = [
            {
                id: 'diary_001',
                content: '今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。',
                date: '2026-01-10',
                mood: 'happy',
                tags: ['生活', '散步']
            },
            {
                id: 'diary_002',
                content: '今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。',
                date: '2026-01-11',
                mood: 'excited',
                tags: ['工作', '成就']
            },
            {
                id: 'diary_003',
                content: '周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。',
                date: '2026-01-12',
                mood: 'relaxed',
                tags: ['户外', '朋友']
            },
            {
                id: 'diary_004',
                content: '今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。',
                date: '2026-01-12',
                mood: 'curious',
                tags: ['学习', '技术']
            },
            {
                id: 'diary_005',
                content: '晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。',
                date: '2026-01-13',
                mood: 'proud',
                tags: ['美食', '家庭']
            }
        ];

        // 5. 对每篇日记内容进行向量化并组装数据
        console.log('Generating embeddings...');
        // 使用 Promise.all 并行调用 Embedding 接口，为每一篇日记内容生成 1024 维度的向量
        const diaryData = await Promise.all(
            diaryContents.map(async (diary) => ({
                ...diary,
                vector: await getEmbedding(diary.content) // 将生成的向量附加到 diary 对象的 vector 字段中
            }))
        );

        // 6. 执行数据插入（Insert）
        // 将包含了 id、vector、content、date、mood、tags 的完整数据数组一次性插入 Milvus
        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: diaryData
        });
        console.log(`✓ Inserted ${insertResult.insert_cnt} records\n`);

    } catch (error) {
        console.error('Error:', error.message); // 捕获并打印运行中出现的任何异常错误
    }
}

main(); // 执行主函数