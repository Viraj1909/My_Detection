/** *********************************************************************************
 * @file spscbuffer.h
 * @author Dharmil Shah (dharmil.shah@ishitva.in)
 * @version 0.1
 * @date 2023-07-13
 * 
 * @brief SPSCBuffer is a wait-free single-producer/single-consumer queue 
 * (commonly known as ringbuffer), that provides functionalities 
 * to store and retrive data with proper memory management. \n
 * 
 * - A single producer(push) and a consumer(pop) 
 * can access buffer parallally without any issue.
 * 
 * - More than one producer or consumer may create
 * problem and undefined behavior.
 * 
 * - Reference is taken from a facebook library \b "folly" 
 * https://github.com/facebook/folly/blob/main/folly/ProducerConsumerQueue.h 
 * 
 * Documentation
 * -------------
 * 
 * - Documentation : https://docs.google.com/document/d/1rOnqsF6_JzLwjw2TyQ8s1KR84Tvi2xVNxX8o0UYfoBk/edit?usp=sharing
 * - Class diagram : https://drive.google.com/file/d/1dny2HO711nSAt4g4Qp5oQaa-gdhvxbup/view?usp=drive_link
 * - Flow diagram  : https://drive.google.com/file/d/1z9puvBABYPfCLs8ZIY_RX88uaBByIBxO/view?usp=drive_link
 * 
 * Version history
 * ---------------
 * 
 * \b [v0.1] Initial version
 ***********************************************************************************/

#ifndef SPSCBUFFER_H
#define SPSCBUFFER_H

#include <iostream>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>
#include <functional>           // for callback functions


/**
 * @brief SPSCBuffer is a wait-free single-producer/single-consumer queue 
 * (commonly known as ringbuffer).
 * 
 * @tparam T is the type of data which will be stored into the buffer.
 * It can be object, pointer to object etc.
 * 
 * @warning
 * Only one thread is allowed to push data into the buffer at a time. \n
 * Only one thread is allowed to pop data from the buffer at a time.
 * 
 * - Documentation : https://docs.google.com/document/d/1rOnqsF6_JzLwjw2TyQ8s1KR84Tvi2xVNxX8o0UYfoBk/edit?usp=sharing
 * - Class diagram : https://drive.google.com/file/d/1dny2HO711nSAt4g4Qp5oQaa-gdhvxbup/view?usp=drive_link
 * - Flow diagram  : https://drive.google.com/file/d/1z9puvBABYPfCLs8ZIY_RX88uaBByIBxO/view?usp=drive_link
 */
template <class T>
class SPSCBuffer
{

/*********/
  public:
/*********/

    // Deleting copy constructor and assignment operator
    SPSCBuffer(const SPSCBuffer &) = delete;
    SPSCBuffer &operator=(const SPSCBuffer &) = delete;

    /** 
     * @brief Constructs buffer and initializes members.
     * @param[in] capacity is the maximum capacity of buffer to hold data. 
     * capacity must be >= 2.
     * @param[in] pop_callbackFunction will be called each time 
     * user pops/removes an element from the buffer. \n
     * It is an \b optional parameter and used inside pop() 
     * and ~SPSCBuffer() while removing object from buffer. \n 
     * User can use it for \a free \a memory or anything else. \n
     * 
     * Example
     * -------
     * @code {.cpp}
     * int bufferSize = 6;
     * 
     * // Buffer without callback function
     * SPSCbuffer< Student > buffer1(bufferSize);
     * 
     * // Buffer with callback function
     * auto callBackFunctionToDeleteStudentPointer = [&](Student *student){
     *      student->freeResources();
     *      delete student;
     * };
     * 
     * SPSCBuffer< Student* > buffer2(bufferSize, callBackFunctionToDeleteStudentPointer);
     * @endcode
     * 
     * @note number of usable slots in the buffer at any given time
     * is actually (capacity-1), so if you start with an empty buffer,
     * isFull() will return true after capacity-1 insertions.
     */
    explicit SPSCBuffer(uint32_t capacity, const std::function< void(T&) > &pop_callbackFunction = nullptr)
    :   readIndex_(0),
        writeIndex_(0),
        capacity_(capacity),
        buffer_(static_cast< T* > (std::malloc(sizeof(T) * capacity))),
        callbackFunctionToCallWhilePop(pop_callbackFunction)
    {
        assert(capacity >= 2);
        if (!buffer_){
            throw std::bad_alloc();
        }
    }
 
    /** 
     * @brief Destructor takes care of release the memory occupied by buffer. \n
     * 
     * - If callback function is provided at the time of construction,
     * then it calls that function with each remaning element as paramter in it. \n
     * 
     * - Else if `<T>` is a pointer type, it deallocates the memory 
     * of the objects referenced by those pointers in the buffer. \n
     * 
     * - Otherwise it simply calls destructor for objectes stored into buffer.
     */
    ~SPSCBuffer() noexcept
    {
        // (No real synchronization needed at destructor time: only one thread can be doing this.)

        // If <T> is destructible, free all the remaining data from the buffer
        if (std::is_destructible<T>::value)
        {
            size_t currentRead = readIndex_;
            size_t endIndex = writeIndex_;

            while (currentRead != endIndex)
            {
                if(callbackFunctionToCallWhilePop != nullptr){
                    callbackFunctionToCallWhilePop(buffer_[currentRead]);
                }
                else if constexpr(std::is_pointer<T>::value){
                    T ptr = buffer_[currentRead];
                    delete ptr;
                }
                else {
                    buffer_[currentRead].~T();
                }
                currentRead = (currentRead + 1) % capacity_;
            }
        }

        // Free memory of buffer (dynamic array)
        std::free(buffer_);
    }

    /** 
     * @brief Push the `data` at the end of the buffer.
     * @param[in] data to be pushed into the buffer.
     * @return `true` if data is pushed into buffer successfully, 
     * or `false` if buffer is empty.
     * 
     * ---------
     * 
     * Example
     * -------
     * 
     * @code {.cpp}
     * SPSCBuffer< Studnt* > buffer(5);
     * 
     * Student *ptr = new Student();
     * 
     * if(buffer.push(ptr)) {
     *      cout << "Data pushed";
     * }
     * else {
     *      cout << "Buffer full";
     *      
     *      // User can retry to push after popping data,
     *      // or delete the pointer to avoid memory leakage
     *      delete ptr;
     * }
     * @endcode
     */
    bool push(const T &data) noexcept
    {
        auto const currentWrite = writeIndex_.load(std::memory_order_relaxed);
        auto nextWrite = (currentWrite + 1) % capacity_;
        
        if (nextWrite == readIndex_.load(std::memory_order_acquire)){
            return false; // buffer is full
        }

        new (&buffer_[currentWrite]) T(data);
        writeIndex_.store(nextWrite, std::memory_order_release);
        return true;
    }

    /** 
     * @brief Removes the first element from the buffer.
     * 
     * Deleting element
     * ----------------
     * 
     * - If the callback function is provided at the construction time,
     *   it invokes that callback function with the popped element as its parameter.
     * 
     * - If callback function is not provided and the `<T>` is a pointer type,
     *   this method tries to delete the related object 
     *   with that pointer using `delete`. \n
     * 
     * - If the pointer is not pointing to any valid reference (dangling pointer),
     *   it may lead to the \b segment \b fault or \b crash.
     * 
     * - Else it explicitely calls the destructor[~T()] for that object.
     * 
     * @note
     * If callback function is not provided at construction time, 
     * then it is advisable to use pop(T&)  OR  pop(const std::function< void(T &) > &) 
     * while dealing with pointers since it does not try to delete anything internally,
     * and user can handle it at his/her side.
     * 
     * @code {.cpp}
     * SPSCbuffer< Student > buffer1(5);
     * 
     * if(buffer.pop()){
     *      // Data is popped
     * }
     * else {
     *      // Buffer is empty
     * }
     * @endcode
     */
    bool pop() noexcept
    {
        auto const currentRead = readIndex_.load(std::memory_order_relaxed);
        if(currentRead == writeIndex_.load(std::memory_order_acquire)){
            return false; // empty buffer
        }

        if (callbackFunctionToCallWhilePop != nullptr){
            // invoke callback function with the popped element
            callbackFunctionToCallWhilePop(buffer_[currentRead]);
        }
        else if constexpr(std::is_pointer<T>::value){
            // dereference and delete pointer
            T ptr = buffer_[currentRead];
            if(ptr != nullptr) delete ptr;
        }
        else {
            buffer_[currentRead].~T();
        }

        auto nextRead = (currentRead + 1) % capacity_;
        readIndex_.store(nextRead, std::memory_order_release);
        return true;
    }

    /** 
     * @brief Removes the fist element from the buffer, 
     * and move (or copy) that element into given variable/paramter.
     * @param[out] data is the variable in which the popped element will be stored
     * @return `true` on successfull pop, `false` if buffer is empty
     * 
     * @note
     * Unlike pop(), it does not delete or free the popped data.
     * 
     * ---------
     * 
     * Example
     * -------
     * 
     * @code {.cpp}
     * SPSCbuffer< Student > buffer1(5);
     * 
     * Student student;
     * 
     * if(buffer.pop(student)){
     *      student.doWork();
     *      // Code...
     * }
     * else {
     *      // Buffer is empty
     * }
     * @endcode
     */
    bool pop(T &data) noexcept
    {
        auto const currentRead = readIndex_.load(std::memory_order_relaxed);
        if (currentRead == writeIndex_.load(std::memory_order_acquire)){
            // buffer is empty
            return false;
        }

        data = std::move(buffer_[currentRead]);
        buffer_[currentRead].~T();
        auto nextRead = (currentRead + 1) % capacity_;
        readIndex_.store(nextRead, std::memory_order_release);
        return true;
    }

    /**
     * @brief It pops the first element from the buffer 
     * and calls the `callbackFunction` with the element to be removed as a parameter.
     * @param[in] callbackFunction is called with the element(to be deleted) as a parameter.
     * @return `true` on successful pop, or `false` if buffer is empty.
     * 
     * @warning
     * It does not remove the element like pop() does.
     * User is responsible to remove/free the object inside the callback function.
     * 
     * ------------
     * 
     * Example
     * -------
     * 
     * @code {.cpp}
     * auto callback = [&](Student &data){
     *     // code...
     * }
     * 
     * if(buffer.pop(callback)) {
     *     // Data popped
     * }
     * else {
     *     // Buffer is empty
     * }
     * @endcode
     */
    bool pop(const std::function< void(T&) > &callbackFunction) noexcept
    {
        auto const currentRead = readIndex_.load(std::memory_order_relaxed);
        if(currentRead == writeIndex_.load(std::memory_order_acquire)){
            return false; // empty buffer
        }

        if(callbackFunction) // not nullptr
            callbackFunction(buffer_[currentRead]);
        auto nextRead = (currentRead + 1) % capacity_;
        readIndex_.store(nextRead, std::memory_order_release);
        return true;
    }

    /** 
     * @brief Buffer is empty or not.
     * @return `true` if buffer is empty, or `false`
     */
    bool isEmpty() const noexcept 
    {
        return readIndex_.load(std::memory_order_acquire) ==
                writeIndex_.load(std::memory_order_acquire);
    }

    /** 
     * @brief Buffer is full or not.
     * @return `true` if buffer is full, or `false`
     */
    bool isFull() const noexcept
    {
        auto nextWrite = (writeIndex_.load(std::memory_order_acquire) + 1) % capacity_;
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            // queue is full
            return true;
        }
        return false;
    }

    /** 
     * @brief Get number of elements stored into the buffer.
     * 
     * - If called by consumer, then true size may be more (because producer may
     * be adding items concurrently). \n
     * - If called by producer, then true size may be less (because consumer may
     * be removing items concurrently). \n
     * - It is undefined to call this from any other thread.
     */
    size_t sizeGuess() const noexcept
    {
        int ret = writeIndex_.load(std::memory_order_acquire) -
                    readIndex_.load(std::memory_order_acquire);
        if (ret < 0){
            ret += capacity_;
        }
        return ret;
    }

    /** @brief maximum number of items in the queue. */
    size_t capacity() const noexcept { return capacity_-1; }

/**********/
  private:
/**********/

    /** @brief index poiting to front of the buffer. */
    std::atomic<unsigned int> readIndex_;
    
    /** @brief index poiting to front of the buffer. */
    std::atomic<unsigned int> writeIndex_;

    /** @brief maximum-1 number of elements buffer can store. */
    const uint32_t capacity_;

    /** @brief It is used to point the dynamic array used to store the data. */
    T * const buffer_;

    /** 
     * @brief This callback function is called each time 
     * while removing an element from the buffer.
     * 
     * Return type of the callback function would be \a `void`. \n
     * One and only \a `paramter` it takes is 
     * the element that is to be removed from the buffer,
     * so that user can do whatever he/she 
     * wants to do with that element inside callback function.
     */
    const std::function< void(T&) > callbackFunctionToCallWhilePop;
};



#endif // SPSCBUFFER_H