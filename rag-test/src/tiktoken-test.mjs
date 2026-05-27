import { getEncodingNameForModel, getEncoding } from 'js-tiktoken'

// 1. 根据大模型名称获取其对应的 Token 编码器名称
// 不同的模型（例如 gpt-3.5-turbo, gpt-4, o1 等）使用的分词算法和词表（Encoding）可能不同
const modelName = "gpt-4"

// gpt-4 默认使用 cl100k_base 编码器
const encodingName = getEncodingNameForModel(modelName)

console.log(encodingName)
// 2. 直接根据编码器名称获取 Token 编码器实例
const enc = getEncoding("cl100k_base");
// 3. 测试不同词汇经过 Token 编码器分词后的 Token 数量（Token Length）
// 在大模型中，输入和输出的计费、上下文窗口限制都是以 Token 为基本单位的
// 英文单词通常一个单词占 1 个或几个 Token：
console.log('apple -> Token数:', enc.encode("apple").length);     // 类似常见英文单词，通常为 1 个 Token
console.log('pineapple -> Token数:', enc.encode("pineapple").length); // 复合词可能会被拆分为多个子词，如 pine + apple

// 中文字符在 cl100k_base 编码中，一个汉字通常占用 1 到 3 个 Token：
console.log('苹果 -> Token数:', enc.encode("苹果").length);     // 2个中文字符，占 2 个 Token
console.log('吃饭 -> Token数:', enc.encode("吃饭").length);     // 2个中文字符，占 2 个 Token
console.log('一二三 -> Token数:', enc.encode("一二三").length); // 3个中文字符，可能占 3 到 4 个 Token（根据具体分词合并规则决定）

