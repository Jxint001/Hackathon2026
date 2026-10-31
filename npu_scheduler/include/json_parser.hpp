// 自实现 JSON 数据结构与解析器，读取输入图所需类型，无第三方依赖。
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace npu {
// 表示输入文件、JSON 格式或图数据不符合读取要求。
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// 一个 JSON 值的树形表示，使用类型标签区分有效内容。
struct Json {
    // 枚举类型：JSON 值的六种类别，决定 Json 中哪些字段有效。
    enum Kind { Null, Boolean, Number, String, Array, Object } kind = Null;
    std::string text;
    std::vector<Json> array;
    std::map<std::string, Json> object;
    // 查找对象中的必需字段。
    const Json &at(const std::string &key) const {
        auto i = object.find(key);
        if (kind != Object || i == object.end()) {
            throw InputError("missing JSON field: " + key);
        }
        return i->second;
    }
    // 将 JSON 数字转换为有符号 64 位整数。
    std::int64_t integer() const {
        if (kind != Number || text.find_first_of(".eE") != std::string::npos) {
            throw InputError("expected JSON integer");
        }
        try {
            return std::stoll(text);
        } catch (const std::exception &) {
            throw InputError("integer out of range");
        }
    }
    // 读取 JSON 字符串。
    const std::string &string() const {
        if (kind != String) {
            throw InputError("expected JSON string");
        }
        return text;
    }
    // 读取 JSON 数组。
    const std::vector<Json> &items() const {
        if (kind != Array) {
            throw InputError("expected JSON array");
        }
        return array;
    }
};

// 一个文档对应一个解析器，以游标递归读取 JSON 树。
class JsonParser {
    const std::string &s;
    std::size_t p = 0;
    // 报告当前字节位置的 JSON 语法错误。
    [[noreturn]] void fail() const {
        throw InputError("invalid JSON at byte " + std::to_string(p));
    }
    // 跳过 JSON 允许的空白字符。
    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) {
            ++p;
        }
    }
    // 尝试消费指定字符，匹配前先跳过空白。
    bool take(char c) {
        ws();
        if (p < s.size() && s[p] == c) {
            ++p;
            return true;
        }
        return false;
    }
    // 强制消费指定字符。
    void expect(char c) {
        if (!take(c)) {
            fail();
        }
    }
    // 读取 Unicode 转义中的四位十六进制数。
    unsigned hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            if (p == s.size()) {
                fail();
            }
            char c = s[p++];
            unsigned d;
            if (c >= '0' && c <= '9') {
                d = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                d = c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                d = c - 'A' + 10;
            } else {
                fail();
            }
            v = 16 * v + d;
        }
        return v;
    }
    // 将 Unicode 码点编码为 UTF-8 并追加到字符串。
    static void utf8(std::string &out, unsigned c) {
        if (c < 128) {
            out += static_cast<char>(c);
        } else if (c < 2048) {
            out += static_cast<char>(192 | (c >> 6));
            out += static_cast<char>(128 | (c & 63));
        } else if (c < 65536) {
            out += static_cast<char>(224 | (c >> 12));
            out += static_cast<char>(128 | ((c >> 6) & 63));
            out += static_cast<char>(128 | (c & 63));
        } else {
            out += static_cast<char>(240 | (c >> 18));
            out += static_cast<char>(128 | ((c >> 12) & 63));
            out += static_cast<char>(128 | ((c >> 6) & 63));
            out += static_cast<char>(128 | (c & 63));
        }
    }
    // 读取带双引号的字符串并处理 JSON 转义及代理对。
    std::string str() {
        expect('"');
        std::string out;
        while (p < s.size()) {
            unsigned char c = static_cast<unsigned char>(s[p++]);
            if (c == '"') {
                return out;
            }
            if (c < 32) {
                fail();
            }
            if (c != '\\') {
                out += static_cast<char>(c);
                continue;
            }
            if (p == s.size()) {
                fail();
            }
            switch (s[p++]) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (p + 2 > s.size() || s[p++] != '\\' || s[p++] != 'u') {
                        fail();
                    }
                    unsigned lo = hex4();
                    if (lo < 0xdc00 || lo > 0xdfff) {
                        fail();
                    }
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    fail();
                }
                utf8(out, cp);
                break;
            }
            default:
                fail();
            }
        }
        fail();
    }
    // 递归解析一个 JSON 值，限制嵌套深度。
    Json value(unsigned depth) {
        if (depth > 128) {
            fail();
        }
        ws();
        if (p == s.size()) {
            fail();
        }
        Json v;
        if (s[p] == '{') {
            ++p;
            v.kind = Json::Object;
            if (take('}')) {
                return v;
            }
            do {
                auto k = str();
                expect(':');
                auto x = value(depth + 1);
                if (!v.object.emplace(std::move(k), std::move(x)).second) {
                    fail();
                }
            } while (take(','));
            expect('}');
        } else if (s[p] == '[') {
            ++p;
            v.kind = Json::Array;
            if (take(']')) {
                return v;
            }
            do {
                v.array.push_back(value(depth + 1));
            } while (take(','));
            expect(']');
        } else if (s[p] == '"') {
            v.kind = Json::String;
            v.text = str();
        } else if (s.compare(p, 4, "null") == 0) {
            p += 4;
        } else if (s.compare(p, 4, "true") == 0) {
            p += 4;
            v.kind = Json::Boolean;
            v.text = "true";
        } else if (s.compare(p, 5, "false") == 0) {
            p += 5;
            v.kind = Json::Boolean;
            v.text = "false";
        } else {
            std::size_t start = p;
            if (s[p] == '-') {
                ++p;
            }
            if (p == s.size() || !std::isdigit(static_cast<unsigned char>(s[p]))) {
                fail();
            }
            if (s[p] == '0') {
                ++p;
            } else {
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
                    ++p;
                }
            }
            if (p < s.size() && s[p] == '.') {
                ++p;
                std::size_t b = p;
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
                    ++p;
                }
                if (b == p) {
                    fail();
                }
            }
            if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
                ++p;
                if (p < s.size() && (s[p] == '+' || s[p] == '-')) {
                    ++p;
                }
                std::size_t b = p;
                while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
                    ++p;
                }
                if (b == p) {
                    fail();
                }
            }
            v.kind = Json::Number;
            v.text = s.substr(start, p - start);
        }
        return v;
    }

  public:
    // 绑定待解析文本，初始化游标。
    explicit JsonParser(const std::string &input) : s(input) {}
    // 解析完整 JSON 文档并检查尾部无多余内容。
    Json parse() {
        auto v = value(0);
        ws();
        if (p != s.size()) {
            fail();
        }
        return v;
    }
};
}
