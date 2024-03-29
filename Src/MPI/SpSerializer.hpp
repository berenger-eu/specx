// Copyright (c) 2018 the complexes++ development team and contributors
// (see the file AUTHORS for the full list of names)
//
// This file is part of complexes++.
//
// complexes++ is free software: you can redistribute it and/or modify
// it under the terms of the Lesser GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// complexes++ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with complexes++.  If not, see <https://www.gnu.org/licenses/>

///////////////////////////////////////////////////////////////////////////
// This is a slightly modified version of the src/io/serializer.h file of the complexes++ project.
// https://github.com/bio-phys/complexespp
//
// [10 Aug 2022] The differences to the source from the complexes++ project are minor:
//	1. Removed namespace "io"
// 	2. Changed class names [prefixed with Sp]
//	3. Replace DEBUG_ASSERT with SP_DEBUG_ASSERT
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#ifndef SPSERIALIZER_HPP
#define SPSERIALIZER_HPP
#define FMT_HEADER_ONLY

#include <array>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "Utils/SpDebugAssert.hpp"

template <typename T, typename O> struct IsConvertible {
  template <typename U>
  static auto test(O *p) -> decltype(U(*p), std::true_type());

  template <typename U> static std::false_type test(...);

  static const bool value =
      std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
};

class SpSerializer;

template <typename, typename = std::void_t<>>
struct class_has_serialize_method
: public std::false_type {};

template <typename Class>
struct class_has_serialize_method<Class,
    std::void_t<decltype(std::declval<Class>().serialize(std::declval<SpSerializer&>()))>>
: public std::is_same<decltype(std::declval<Class>().serialize(std::declval<SpSerializer&>())), void>
{};

class SpSerializer {
  std::vector<unsigned char> m_buffer;

  void appendKey(const std::string &inKey) {
    size_t keylength = inKey.length();
    m_buffer.insert(
        m_buffer.end(), reinterpret_cast<const unsigned char *>(&keylength),
        reinterpret_cast<const unsigned char *>(&keylength) + sizeof(size_t));
    m_buffer.insert(m_buffer.end(),
                    reinterpret_cast<const unsigned char *>(inKey.c_str()),
                    reinterpret_cast<const unsigned char *>(inKey.c_str()) +
                        inKey.length());
  }

public:
  SpSerializer() {}

  void append(const unsigned char array[], const size_t size,
              const std::string &inKey) {
    appendKey(inKey);

    m_buffer.insert(
        m_buffer.end(), reinterpret_cast<const unsigned char *>(&size),
        reinterpret_cast<const unsigned char *>(&size) + sizeof(size_t));

    m_buffer.insert(m_buffer.end(), &array[0], &array[size]);

    m_buffer.push_back(static_cast<unsigned char>(~0));
  }

  /////////////////////////////////////////////////////////////////////
  /// POD
  /////////////////////////////////////////////////////////////////////

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const ItemClass array[], const size_t nbItems,
              const std::string &inKey) {
    append(nbItems, inKey + "-size");
    append(reinterpret_cast<const unsigned char *>(&array[0]),
           sizeof(ItemClass) * nbItems, inKey);
  }

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const ItemClass &item, const std::string &inKey) {
    append(reinterpret_cast<const unsigned char *>(&item), sizeof(ItemClass),
           inKey);
  }

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const std::vector<ItemClass> &array, const std::string &inKey) {
    append(array.size(), inKey + "-size");
    append(reinterpret_cast<const unsigned char *>(&array[0]),
           array.size() * sizeof(ItemClass), inKey);
  }

  template <
      class ItemClass, std::size_t N,
      typename std::enable_if<std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const std::array<ItemClass, N> &array, const std::string &inKey) {
    append(N, inKey + "-size");
    append(reinterpret_cast<const unsigned char *>(&array[0]),
           N * sizeof(ItemClass), inKey);
  }

  /////////////////////////////////////////////////////////////////////
  ///  NOT POD
  /////////////////////////////////////////////////////////////////////

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const ItemClass &item, const std::string &inKey) {
    static_assert(class_has_serialize_method<ItemClass>::value,
        "Class must inherit from SpAbstractSerializable");
    appendKey(inKey);
    item.serialize(*this);
  }

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const ItemClass array[], const size_t nbItems,
              const std::string &inKey) {
    static_assert(class_has_serialize_method<ItemClass>::value,
        "Class must inherit from SpAbstractSerializable");

    append(nbItems, inKey + "-size");
    for (size_t idx = 0; idx < nbItems; ++idx) {
      appendKey(inKey + std::to_string(idx));
      array[idx].serialize(*this);
    }
  }

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const std::vector<ItemClass> &array, const std::string &inKey) {
    append(array.size(), inKey + "-size");
    for (size_t idx = 0; idx < array.size(); ++idx) {
      append(array[idx], inKey + std::to_string(idx));
    }
  }

  template <
      class ItemClass, std::size_t N,
      typename std::enable_if<!std::is_pod<ItemClass>::value, int>::type = 0>
  void append(const std::array<ItemClass, N> &array, const std::string &inKey) {
    append(N, inKey + "-size");
    for (size_t idx = 0; idx < N; ++idx) {
      append(array[idx], inKey + std::to_string(idx));
    }
  }

  /////////////////////////////////////////////////////////////////////
  /// Utils
  /////////////////////////////////////////////////////////////////////

  template <class ItemClass>
  void appendStreamed(const ItemClass &item, const std::string &inKey) {
    std::stringstream stream;
    stream << item;
    append(stream.str(), inKey);
  }

  void append(const std::string &str, const std::string &inKey) {
    append(static_cast<size_t>(str.length()), inKey + "-length");
    append(reinterpret_cast<const unsigned char *>(str.c_str()), str.length(),
           inKey);
  }

  const std::vector<unsigned char> &getBuffer() const { return m_buffer; }

  std::vector<unsigned char> releaseBuffer() { return std::move(m_buffer); }
};

class SpDeserializer {
  const unsigned char *m_buffer;
  const size_t m_bufferSize;

  size_t m_currentIndex;

public:
  SpDeserializer(const unsigned char *inBuffer, const size_t inBufferSize)
      : m_buffer(inBuffer), m_bufferSize(inBufferSize), m_currentIndex(0) {}

  SpDeserializer &access(const std::string &inKey) {
    SP_DEBUG_ASSERT(m_currentIndex + sizeof(size_t) <= m_bufferSize,
                 "Unpack too much data for key " + inKey);

    size_t keySize;
    std::copy(&m_buffer[m_currentIndex],
              &m_buffer[m_currentIndex + sizeof(size_t)],
              reinterpret_cast<unsigned char *>(&keySize));
    m_currentIndex += sizeof(size_t);

    SP_DEBUG_ASSERT(m_currentIndex + keySize <= m_bufferSize,
                 "Unpack too much data for key " + inKey);

    std::vector<char> recvKey(keySize + 1);
    std::copy(&m_buffer[m_currentIndex], &m_buffer[m_currentIndex + keySize],
              reinterpret_cast<unsigned char *>(recvKey.data()));
    m_currentIndex += keySize;
    recvKey[keySize] = '\0';
    const std::string recvKeyStr = recvKey.data();

    SP_DEBUG_ASSERT(inKey == recvKeyStr, "Key missmatch, asked " + inKey + " but it is " + recvKeyStr);

    return *this;
  }

  void restore(unsigned char array[], const size_t size,
               const std::string &inKey) {
    access(inKey);

    size_t nextPackSize;
    std::copy(&m_buffer[m_currentIndex],
              &m_buffer[m_currentIndex + sizeof(size_t)],
              reinterpret_cast<unsigned char *>(&nextPackSize));
    m_currentIndex += sizeof(size_t);
    SP_DEBUG_ASSERT(
        nextPackSize == size,
        "Next message is of size " + std::to_string(nextPackSize) + " and it has been asked for " + std::to_string(size) + " key " + inKey);

    SP_DEBUG_ASSERT(m_currentIndex + size <= m_bufferSize, "Unpack too much data");
    std::copy(&m_buffer[m_currentIndex], &m_buffer[m_currentIndex + size],
              array);
    m_currentIndex += size;

    SP_DEBUG_ASSERT(m_currentIndex + 1 <= m_bufferSize, "Unpack too much data");
    SP_DEBUG_ASSERT(m_buffer[m_currentIndex] == static_cast<unsigned char>(~0),
                 "Invalid check value, should be " +std::to_string(static_cast<unsigned char>(~0))
                    + " is " + std::to_string(m_buffer[m_currentIndex])
                    + ", for key " + inKey);
    m_currentIndex += 1;
  }

  ///////////////////////////////////////////////////////////////////////
  /// POD
  ///////////////////////////////////////////////////////////////////////

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  size_t restore(ItemClass *&array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    delete[] array;
    array = new ItemClass[nbItems];
    restore(reinterpret_cast<unsigned char *>(&array[0]),
            sizeof(ItemClass) * nbItems, inKey);
    return nbItems;
  }

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(ItemClass array[], const size_t nbItems,
               const std::string &inKey) {
    const size_t nbItemsWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(nbItems == nbItemsWritten, "Invalid number of elements");
    restore(reinterpret_cast<unsigned char *>(&array[0]),
            sizeof(ItemClass) * nbItems, inKey);
  }

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(ItemClass &item, const std::string &inKey) {
    restore(reinterpret_cast<unsigned char *>(&item), sizeof(ItemClass), inKey);
  }

  template <class ItemClass, typename std::enable_if<
                                 std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(std::vector<ItemClass> &array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    array.resize(nbItems);
    restore(reinterpret_cast<unsigned char *>(&array[0]),
            array.size() * sizeof(ItemClass), inKey);
  }

  template <
      class ItemClass, std::size_t N,
      typename std::enable_if<std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(std::array<ItemClass, N> &array, const std::string &inKey) {
    const size_t NWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(N == NWritten, "Invalid number of elements");
    restore(reinterpret_cast<unsigned char *>(&array[0]), N * sizeof(ItemClass),
            inKey);
  }

  ///////////////////////////////////////////////////////////////////////
  /// Not POD
  ///////////////////////////////////////////////////////////////////////

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  size_t restore(ItemClass *&array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    delete[] array;
    array = reinterpret_cast<ItemClass *>(
        new unsigned char[sizeof(ItemClass) * nbItems]);
    for (size_t idx = 0; idx < nbItems; ++idx) {
      access(inKey + std::to_string(idx));
      new (&array[idx]) ItemClass(*this);
    }
    return nbItems;
  }

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(ItemClass array[], const size_t nbItems,
               const std::string &inKey) {
    const size_t nbItemsWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(nbItems == nbItemsWritten, "Invalid number of elements");
    for (size_t idx = 0; idx < nbItems; ++idx) {
      access(inKey + std::to_string(idx));
      array[idx] = ItemClass(*this);
    }
  }

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(ItemClass &item, const std::string &inKey) {
    access(inKey);
    item = ItemClass(*this);
  }

  template <class ItemClass, typename std::enable_if<
                                 !std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(std::vector<ItemClass> &array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    array.clear();
    array.reserve(nbItems);
    for (size_t idx = 0; idx < nbItems; ++idx) {
      access(inKey + std::to_string(idx));
      array.emplace_back(*this);
    }
  }

  template <
      class ItemClass, std::size_t N,
      typename std::enable_if<!std::is_pod<ItemClass>::value, int>::type = 0>
  void restore(std::array<ItemClass, N> &array, const std::string &inKey) {
    const size_t NWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(N == NWritten, "Invalid number of elements");
    for (size_t idx = 0; idx < N; ++idx) {
      access(inKey + std::to_string(idx));
      restore(array[idx], inKey + std::to_string(idx));
    }
  }

  ///////////////////////////////////////////////////////////////////////
  /// String
  ///////////////////////////////////////////////////////////////////////

  size_t restore(std::string *&array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    delete[] array;
    array = new std::string[nbItems];
    for (size_t idx = 0; idx < nbItems; ++idx) {
      restore(array[idx], inKey + std::to_string(idx));
    }
    return nbItems;
  }

  void restore(std::string array[], const size_t nbItems,
               const std::string &inKey) {
    const size_t nbItemsWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(nbItems == nbItemsWritten, "Invalid number of elements");
    for (size_t idx = 0; idx < nbItems; ++idx) {
      restore(array[idx], inKey + std::to_string(idx));
    }
  }

  void restore(std::vector<std::string> &array, const std::string &inKey) {
    const size_t nbItems = restore<size_t>(inKey + "-size");
    array.clear();
    array.resize(nbItems);
    for (size_t idx = 0; idx < array.size(); ++idx) {
      restore(array[idx], inKey + std::to_string(idx));
    }
  }

  template <std::size_t N>
  void restore(std::array<std::string, N> &array, const std::string &inKey) {
    const size_t NWritten = restore<size_t>(inKey + "-size");
    SP_DEBUG_ASSERT(N == NWritten, "Invalid number of elements");
    for (size_t idx = 0; idx < N; ++idx) {
      restore(array[idx], inKey + std::to_string(idx));
    }
  }

  ///////////////////////////////////////////////////////////////////////
  /// Extra
  ///////////////////////////////////////////////////////////////////////

  template <class ItemClass>
  void restoreStreamed(ItemClass &item, const std::string &inKey) {
    std::string str;
    restore(str, inKey);
    std::stringstream stream(str);
    stream >> item;
  }

  void restore(std::string &str, const std::string &inKey) {
    const size_t length = restore<size_t>(inKey + "-length");
    std::vector<char> tmpStr(length);
    restore(reinterpret_cast<unsigned char *>(tmpStr.data()), length, inKey);
    tmpStr.push_back('\0');
    str = tmpStr.data();
  }

  template <class ItemClass,
            typename std::enable_if<
                !IsConvertible<ItemClass, SpDeserializer>::value, int>::type = 0>
  ItemClass restore(const std::string &inKey) {
    typename std::remove_const<ItemClass>::type item;
    restore(item, inKey);
    return item;
  }

  template <class ItemClass,
            typename std::enable_if<
                IsConvertible<ItemClass, SpDeserializer>::value, int>::type = 0>
  ItemClass restore(const std::string &inKey) {
    access(inKey);
    return typename std::remove_const<ItemClass>::type(*this);
  }

  template <class ItemClass>
  ItemClass restoreStreamed(const std::string &inKey) {
    typename std::remove_const<ItemClass>::type item;
    restoreStreamed(item, inKey);
    return item;
  }
};

#endif
