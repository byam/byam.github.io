---
layout: post
title:  "A Tour of Go"
date:   2018-04-25 00:00:00 +0900
categories: go,preference
fbcomments: true
---

Outline:

- [Flow control statements](#flow-control-statements)
    - [For](#for)
    - [If](#if)
    - [Swtich](#switch)
    - [Defer](#defer)
- [More types](#more-types)
    - [Pointers](#pointers)
    - [Structs](#structs)
    - [Arrays](#arrays)
    - [Slices](#slices)
    - [Range](#range)
    - [Maps](#maps)
    - [Function values](#function-values)
- [Method and Interfaces](#method-and-interfaces)
    - [Methods](#methods)
    - [Interfaces](#interfaces)
- [Concurrency](#concurrency)

## Flow control statements

### For

```go
package main

import "fmt"

func main() {
	sum := 0
	for i := 0; i < 10; i++ {
		sum += i
	}
	fmt.Println(sum)
}
```

- there are **no parentheses** surrounding the three components of the for statement
- the braces `{ }` are always required.
- The init and post statement are optional.
    ```go
    for ; sum < 1000; {
        sum += sum
    }
    ```
- `for` is Go's `while`
    ```go
	for sum < 1000 {
		sum += sum
	}
    ```
- forever
    ```go
	for {
	}
    ```

### If

```go
package main

import (
	"fmt"
	"math"
)

func sqrt(x float64) string {
	if x < 0 {
		return sqrt(-x) + "i"
	}
	return fmt.Sprint(math.Sqrt(x))
}

func main() {
	fmt.Println(sqrt(2), sqrt(-4))
}
```

- If with a short statement
    ```go
    func pow(x, n, lim float64) float64 {
        if v := math.Pow(x, n); v < lim {
            return v
        }
        return lim
    }
    ```
- If and else
    ```go
    func pow(x, n, lim float64) float64 {
        if v := math.Pow(x, n); v < lim {
            return v
        } else {
            fmt.Printf("%g >= %g\n", v, lim)
        }
        // can't use v here, though
        return lim
    }
    ```

## Switch

```go
package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Print("Go runs on ")
	switch os := runtime.GOOS; os {
	case "darwin":
		fmt.Println("OS X.")
	case "linux":
		fmt.Println("Linux.")
	default:
		// freebsd, openbsd,
		// plan9, windows...
		fmt.Printf("%s.", os)
	}
}
```

- the `break` statement that is needed at the end of each case in those languages is provided automatically in Go.
- Switch cases evaluate cases from top to bottom, stopping when a case succeeds.
- Switch without a condition is the same as `switch true`.
    ```go
    package main
    
    import (
        "fmt"
        "time"
    )
    
    func main() {
        t := time.Now()
        switch {
        case t.Hour() < 12:
            fmt.Println("Good morning!")
        case t.Hour() < 17:
            fmt.Println("Good afternoon.")
        default:
            fmt.Println("Good evening.")
        }
    }
    ```
    
### Defer

- A `defer` statement defers the execution of a function until the surrounding function returns.
- The deferred call's arguments are evaluated immediately, but the function call is not executed until the surrounding function returns.

```go
package main

import "fmt"

func main() {
	defer fmt.Println("world")

	fmt.Println("hello")
}
```
Output:
```bash
hello
world
```

#### Stacking defers
- Deferred function calls are pushed onto a `stack`. When a function returns, its deferred calls are executed in `last-in-first-out` order.

```go
package main

import "fmt"

func main() {
	fmt.Println("counting")

	for i := 0; i < 10; i++ {
		defer fmt.Println(i)
	}

	fmt.Println("done")
}
```


## More Types

### Pointers

- Go has pointers. A pointer holds the memory address of a value.

```go
package main

import "fmt"

func main() {
	i, j := 42, 2701

	p := &i         // point to i
	fmt.Println(*p) // read i through the pointer
	*p = 21         // set i through the pointer
	fmt.Println(i)  // see the new value of i

	p = &j         // point to j
	*p = *p / 37   // divide j through the pointer
	fmt.Println(j) // see the new value of j
}

```

### Structs

```go
package main

import "fmt"

type Vertex struct {
	X int
	Y int
}

func main() {
	v := Vertex{1, 2}
	v.X = 4
	fmt.Println(v.X)
}
```

- A `struct` is a collection of fields.
- Struct fields are accessed using a dot.
- Struct fields can be accessed through a struct pointer.
- Struct Literals
    ```go
    package main
    
    import "fmt"
    
    type Vertex struct {
        X, Y int
    }
    
    var (
        v1 = Vertex{1, 2}  // has type Vertex
        v2 = Vertex{X: 1}  // Y:0 is implicit
        v3 = Vertex{}      // X:0 and Y:0
        p  = &Vertex{1, 2} // has type *Vertex
    )
    
    func main() {
        fmt.Println(v1, p, v2, v3)
    }
    ```

### Arrays

```go
package main

import "fmt"

func main() {
	var a [2]string
	a[0] = "Hello"
	a[1] = "World"
	fmt.Println(a[0], a[1])
	fmt.Println(a)

	primes := [6]int{2, 3, 5, 7, 11, 13}
	fmt.Println(primes)
}
```

### Slices

- An `array` has a **fixed size**. 
- A `slice`, on the other hand, is a **dynamically-sized**, flexible view into the elements of an array. 
- In practice, slices are much more common than arrays.

```go
package main

import "fmt"

func main() {
	primes := [6]int{2, 3, 5, 7, 11, 13}

	var s []int = primes[1:4]
	fmt.Println(s)
}
```

#### Slices are like references to arrays
- A slice **does not store any data**, it just describes a section of an **underlying array**.
- Changing the elements of a slice modifies the corresponding elements of its underlying array.
- Other slices that share the same underlying array will see those changes.

```go
package main

import "fmt"

func main() {
	names := [4]string{
		"John",
		"Paul",
		"George",
		"Ringo",
	}
	fmt.Println(names)

	a := names[0:2]
	b := names[1:3]
	fmt.Println(a, b)

	b[0] = "XXX"
	fmt.Println(a, b)
	fmt.Println(names)
}
```

Output:
```bash
[John Paul George Ringo]
[John Paul] [Paul George]
[John XXX] [XXX George]
[John XXX George Ringo]
```

#### Slice length and capacity
- A slice has both a **length** and a **capacity**.
- The length of a slice is the number of elements it contains.
- The capacity of a slice is the number of elements in the underlying array, counting from the first element in the slice.
- The length and capacity of a slice s can be obtained using the expressions `len(s)` and `cap(s)`.

```go
package main

import "fmt"

func main() {
	s := []int{2, 3, 5, 7, 11, 13}
	printSlice(s)

	// Slice the slice to give it zero length.
	s = s[:0]
	printSlice(s)

	// Extend its length.
	s = s[:4]
	printSlice(s)

	// Drop its first two values.
	s = s[2:]
	printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}
```

Output:
```bash
len=6 cap=6 [2 3 5 7 11 13]
len=0 cap=6 []
len=4 cap=6 [2 3 5 7]
len=2 cap=4 [5 7]
```

- Creating a slice with make
    ```go
    b := make([]int, 0, 5) // len(b)=0, cap(b)=5
    
    b = b[:cap(b)] // len(b)=5, cap(b)=5
    b = b[1:]      // len(b)=4, cap(b)=4
    ```

#### Creating a slice with make

- Slices can be created with the built-in `make` function; this is how you create **dynamically-sized arrays**.
- The `make` function allocates a **zeroed array** and returns a slice that refers to that array:

```go
package main

import "fmt"

func main() {
	a := make([]int, 5)
	printSlice("a", a)

	b := make([]int, 0, 5)
	printSlice("b", b)

	c := b[:2]
	printSlice("c", c)

	d := c[2:5]
	printSlice("d", d)
}

func printSlice(s string, x []int) {
	fmt.Printf("%s len=%d cap=%d %v\n",
		s, len(x), cap(x), x)
}
```

```bash
a len=5 cap=5 [0 0 0 0 0]
b len=0 cap=5 []
c len=2 cap=5 [0 0]
d len=3 cap=3 [0 0 0]
```

#### Appending to a slice

```go
package main

import "fmt"

func main() {
	var s []int
	printSlice(s)

	// append works on nil slices.
	s = append(s, 0)
	printSlice(s)

	// The slice grows as needed.
	s = append(s, 1)
	printSlice(s)

	// We can add more than one element at a time.
	s = append(s, 2, 3, 4)
	printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}

```

Output:
```bash
len=0 cap=0 []
len=1 cap=1 [0]
len=2 cap=2 [0 1]
len=5 cap=6 [0 1 2 3 4]
```

### Range

- The `range` form of the for loop iterates over a `slice` or `map`.
- When ranging over a `slice`, two values are returned for each iteration. 
    - The first is the index, and the second is a copy of the element at that index.

```go
package main

import "fmt"

var pow = []int{1, 2, 4, 8, 16, 32, 64, 128}

func main() {
	for i, v := range pow {
		fmt.Printf("2**%d = %d\n", i, v)
	}
}
```


- You can skip the index or value by assigning to `_`.

### Maps

- A `map` maps keys to values.
- The zero value of a map is `nil`. A nil map has no keys, nor can keys be added.
- The make function returns a `map` of the given type, initialized and ready for use.

```go
package main

import "fmt"

type Vertex struct {
	Lat, Long float64
}

var m = map[string]Vertex{
	"Bell Labs": {
		40.68433, -74.39967,
	},
	"Google": {
		37.42202, -122.08408,
	},
}

func main() {
	fmt.Println(m)
}
```

#### Mutating Maps

```go
package main

import "fmt"

func main() {
	m := make(map[string]int)

	m["Answer"] = 42
	fmt.Println("The value:", m["Answer"])

	m["Answer"] = 48
	fmt.Println("The value:", m["Answer"])

	delete(m, "Answer")
	fmt.Println("The value:", m["Answer"])

	v, ok := m["Answer"]
	fmt.Println("The value:", v, "Present?", ok)
}
```

Output:
```bash
The value: 42
The value: 48
The value: 0
The value: 0 Present? false
```


### Function values

- Functions are values too. They can be passed around just like other values.
- Function values may be used as function arguments and return values.

```go
package main

import (
	"fmt"
	"math"
)

func compute(fn func(float64, float64) float64) float64 {
	return fn(3, 4)
}

func main() {
	hypot := func(x, y float64) float64 {
		return math.Sqrt(x*x + y*y)
	}
	fmt.Println(hypot(5, 12))

	fmt.Println(compute(hypot))
	fmt.Println(compute(math.Pow))
}

```

Output:
```bash
13
5
81
```

#### Function closures

- Go functions may be closures. 
- A closure is a function value that references variables from outside its body. 
- The function may access and assign to the referenced variables; in this sense the function is "bound" to the variables.
- For example, the `adder` function returns a closure. Each closure is bound to its own sum variable.

```go
package main

import "fmt"

func adder() func(int) int {
	sum := 0
	return func(x int) int {
		sum += x
		return sum
	}
}

func main() {
	pos, neg := adder(), adder()
	for i := 0; i < 10; i++ {
		fmt.Println(
			pos(i),
			neg(-2*i),
		)
	}
}
```

Output:
```bash
0 0
1 -2
3 -6
6 -12
10 -20
15 -30
21 -42
28 -56
36 -72
45 -90
```



## Method and Interfaces

### Methods

- Go does not have classes. However, you can define methods on types.
- A method is a function with a special `receiver` argument.
- The `receiver` appears in its own argument list between the func keyword and the method name.
- In this example, the `Abs` method has a `receiver` of type `Vertex` named `v`.

```go
package main

import (
	"fmt"
	"math"
)

type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := Vertex{3, 4}
	fmt.Println(v.Abs())
}
```

- Remember: a method is just a function with a receiver argument.
- You can declare a method on non-struct types, too.

```go
package main

import (
	"fmt"
	"math"
)

type MyFloat float64

func (f MyFloat) Abs() float64 {
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

func main() {
	f := MyFloat(-math.Sqrt2)
	fmt.Println(f.Abs())
}
```

#### Pointer receivers

- You can declare methods with **pointer receivers**.
- Methods with pointer receivers **can modify the value** to which the receiver points 
- Since methods often need to modify their receiver, **pointer receivers are more common** than value receivers.

```go
package main

import (
	"fmt"
	"math"
)

type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v *Vertex) Scale(f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func main() {
	v := Vertex{3, 4}
	v.Scale(10)
	fmt.Println(v.Abs())
}
```

- With a **value receiver**, the Scale method operates on a copy of the original Vertex value. 
    - (This is the same behavior as for any other function argument.) 
- The Scale method must have a **pointer receiver** to change the `Vertex` value declared in the `main` function.



#### Choosing a value or pointer receiver

- There are two reasons to use a pointer receiver.
    - The first is so that the method **can modify the value** that its receiver points to.
    - The second is to **avoid copying the value** on each method call. 
        - This can be more efficient if the receiver is a large struct, for example.

- In general, all methods on a given type should have either value or pointer receivers, but not a mixture of both.

```go
package main

import (
	"fmt"
	"math"
)

type Vertex struct {
	X, Y float64
}

func (v *Vertex) Scale(f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func (v *Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := &Vertex{3, 4}
	fmt.Printf("Before scaling: %+v, Abs: %v\n", v, v.Abs())
	v.Scale(5)
	fmt.Printf("After scaling: %+v, Abs: %v\n", v, v.Abs())
}
```

### Interfaces

- An `interface` type is defined as a set of method signatures.
- A value of interface type can hold any value that implements those methods.

```go
package main

import (
	"fmt"
	"math"
)

type Abser interface {
	Abs() float64
}

func main() {
	var a Abser
	f := MyFloat(-math.Sqrt2)
	v := Vertex{3, 4}

	a = f  // a MyFloat implements Abser
	a = &v // a *Vertex implements Abser

	fmt.Println(a.Abs())
}

type MyFloat float64

func (f MyFloat) Abs() float64 {
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

type Vertex struct {
	X, Y float64
}

func (v *Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
```

#### Interface values

- Under the covers, interface values can be thought of as a tuple of a **value** and a **concrete type**

```go
package main

import (
	"fmt"
	"math"
)

type I interface {
	M()
}

type T struct {
	S string
}

func (t *T) M() {
	fmt.Println(t.S)
}

type F float64

func (f F) M() {
	fmt.Println(f)
}

func main() {
	var i I

	i = &T{"Hello"}
	describe(i)
	i.M()

	i = F(math.Pi)
	describe(i)
	i.M()
}

func describe(i I) {
	fmt.Printf("(%v, %T)\n", i, i)
}
```

Output:
```bash
(&{Hello}, *main.T)
Hello
(3.141592653589793, main.F)
3.141592653589793
```


#### Type assertions

- A type `assertion` provides access to an interface value's underlying concrete value.

```go
package main

import "fmt"

func main() {
	var i interface{} = "hello"

	s := i.(string)
	fmt.Println(s)

	s, ok := i.(string)
	fmt.Println(s, ok)

	f, ok := i.(float64)
	fmt.Println(f, ok)

	f = i.(float64) // panic
	fmt.Println(f)
}
```

Output:
```bash
hello
hello true
0 false
panic: interface conversion: interface {} is string, not float64

```

#### Stringers

```go
package main

import "fmt"

type Person struct {
	Name string
	Age  int
}

func (p Person) String() string {
	return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
}

func main() {
	a := Person{"Arthur Dent", 42}
	z := Person{"Zaphod Beeblebrox", 9001}
	fmt.Println(a, z)
}
```

Output:
```bash
Arthur Dent (42 years) Zaphod Beeblebrox (9001 years)
```


#### Errors

```go
package main

import (
	"fmt"
	"time"
)

type MyError struct {
	When time.Time
	What string
}

func (e *MyError) Error() string {
	return fmt.Sprintf("at %v, %s",
		e.When, e.What)
}

func run() error {
	return &MyError{
		time.Now(),
		"it didn't work",
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
	}
}
```

Output:
```bash
at 2018-04-25 17:29:12.80395766 +0900 JST m=+0.000299919, it didn't work
```


#### Readers

```go
package main

import (
	"fmt"
	"io"
	"strings"
)

func main() {
	r := strings.NewReader("Hello, Reader!")

	b := make([]byte, 8)
	for {
		n, err := r.Read(b)
		fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
		fmt.Printf("b[:n] = %q\n", b[:n])
		if err == io.EOF {
			break
		}
	}
}
```

Output:
```bash
n = 8 err = <nil> b = [72 101 108 108 111 44 32 82]
b[:n] = "Hello, R"
n = 6 err = <nil> b = [101 97 100 101 114 33 32 82]
b[:n] = "eader!"
n = 0 err = EOF b = [101 97 100 101 114 33 32 82]
b[:n] = ""
```

## Concurrency

### Goroutines
- A `goroutine` is a lightweight thread managed by the Go runtime.

```go
package main

import (
	"fmt"
	"time"
)

func say(s string) {
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		fmt.Println(s)
	}
}

func main() {
	go say("world")
	say("hello")
}
```

Output:
```bash
world
hello
hello
world
world
hello
world
hello
world
hello
```

### Channels

- Channels are a typed conduit through which you can **send** and **receive** values with the channel operator, `<-`
    ```go
    ch <- v    // Send v to channel ch.
    v := <-ch  // Receive from ch, and
               // assign value to v.
    ```
- Like maps and slices, channels must be created before use:
    ```go
    ch := make(chan int)
    ```
- By default, sends and receives block until the other side is ready. 
- This allows goroutines to synchronize without explicit locks or condition variables.

```go
package main

import "fmt"

func sum(s []int, c chan int) {
	sum := 0
	for _, v := range s {
		sum += v
	}
	c <- sum // send sum to c
}

func main() {
	s := []int{7, 2, 8, -9, 4, 0}

	c := make(chan int)
	go sum(s[:len(s)/2], c)
	go sum(s[len(s)/2:], c)
	x, y := <-c, <-c // receive from c

	fmt.Println(x, y, x+y)
}
```

Output:
```go
-5 17 12
```


### Buffered Channels

- Channels can be buffered. 
- Provide the buffer length as the second argument to make to initialize a buffered channel:
    `ch := make(chan int, 100)`
- Sends to a buffered channel block only when the buffer is full. 
- Receives block when the buffer is empty.

```go
package main

import "fmt"

func main() {
	ch := make(chan int, 2)
	ch <- 1
	ch <- 2
	fmt.Println(<-ch)
	fmt.Println(<-ch)
}
```

Output:
```bash
1
2
```

### Range and Close

- A sender can close a channel to indicate that no more values will be sent. 
- Receivers can test whether a channel has been closed by assigning a second parameter to the receive expression: after
    - `v, ok := <-ch`
    - `ok` is `false` if there are no more values to receive and the channel is closed.
- The loop `for i := range c` receives values from the channel repeatedly until it is closed.
- Only the sender should close a channel, never the receiver. Sending on a closed channel will cause a panic.
- Channels aren't like files; you don't usually need to close them. Closing is only necessary when the receiver must be told there are no more values coming, such as to terminate a range loop.

```go
package main

import (
	"fmt"
)

func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		c <- x
		x, y = y, x+y
	}
	close(c)
}

func main() {
	c := make(chan int, 10)
	go fibonacci(cap(c), c)
	for i := range c {
		fmt.Println(i)
	}
}
```

Output:
```bash
0
1
1
2
3
5
8
13
21
34
```

### Select

- The `select` statement lets a goroutine wait on multiple communication operations.
- A `select` blocks until one of its cases can run, then it executes that case. It chooses one at random if multiple are ready.

```go
package main

import "fmt"

func fibonacci(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Println("quit")
			return
		}
	}
}

func main() {
	c := make(chan int)
	quit := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-c)
		}
		quit <- 0
	}()
	fibonacci(c, quit)
}
```

Output:
```bash
0
1
1
2
3
5
8
13
21
34
quit
```

#### Default Selection

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	tick := time.Tick(100 * time.Millisecond)
	boom := time.After(500 * time.Millisecond)
	for {
		select {
		case <-tick:
			fmt.Println("tick.")
		case <-boom:
			fmt.Println("BOOM!")
			return
		default:
			fmt.Println("    .")
			time.Sleep(50 * time.Millisecond)
		}
	}
}
```

Output:
```bash
    .
    .
tick.
    .
    .
tick.
    .
    .
tick.
    .
    .
tick.
    .
    .
BOOM!
```

#### Mutex

- This concept is called mutual exclusion, and the conventional name for the data structure that provides it is mutex.
- Go's standard library provides mutual exclusion with `sync.Mutex` and its two methods:
    - `Lock`
    - `Unlock`
- We can define a block of code to be executed in mutual exclusion by surrounding it with a call to `Lock` and `Unlock` as shown on the `Inc` method.
- We can also use `defer` to ensure the `mutex` will be unlocked as in the `Value` method.

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

// SafeCounter is safe to use concurrently.
type SafeCounter struct {
	v   map[string]int
	mux sync.Mutex
}

// Inc increments the counter for the given key.
func (c *SafeCounter) Inc(key string) {
	c.mux.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	c.v[key]++
	c.mux.Unlock()
}

// Value returns the current value of the counter for the given key.
func (c *SafeCounter) Value(key string) int {
	c.mux.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	defer c.mux.Unlock()
	return c.v[key]
}

func main() {
	c := SafeCounter{v: make(map[string]int)}
	for i := 0; i < 1000; i++ {
		go c.Inc("somekey")
	}

	time.Sleep(time.Second)
	fmt.Println(c.Value("somekey"))
}
```

Output:
```bash
1000
```

